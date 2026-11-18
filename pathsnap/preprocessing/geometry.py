"""Vectorized metric GPS/road processing and indexed point-to-segment projection."""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer, Geod
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .config import Config
from .io import DataError

GEOD = Geod(ellps='WGS84')

def valid_lonlat(lon,lat):
    return np.isfinite(lon)&np.isfinite(lat)&(np.abs(lon)<=180)&(np.abs(lat)<=90)

def preprocess_gps(trip, config=None, partition='inference'):
    cfg = (config or Config()).validate()
    arr=trip['track']
    lon,lat,t=arr.T
    n=len(arr); valid=valid_lonlat(lon,lat); tf=np.isfinite(t)
    dt=np.full(n,np.nan)
    dist=np.full(n,np.nan); speed=np.full(n,np.nan)
    bearing=np.full(n,np.nan); accel=np.full(n,np.nan)
    if n>1:
        dt[1:]=np.diff(t)
        ii=np.where(valid[1:]&valid[:-1]&tf[1:]&tf[:-1])[0]+1
        if len(ii):
            az,_,d=GEOD.inv(lon[ii-1],lat[ii-1],lon[ii],lat[ii]); dist[ii]=d
            bearing[ii] = np.mod(az,360)
        good=np.isfinite(dist)&(dt>0)
        speed[good]=dist[good]/dt[good]
        if n>2:
            idx=np.where(np.isfinite(speed[1:])&np.isfinite(speed[:-1])&(dt[1:]>0))[0]+1
            accel[idx]=(speed[idx]-speed[idx-1])/dt[idx]
    flags={
       'invalid_coordinate':~valid, 'invalid_timestamp':~tf,
       'duplicate_timestamp':np.isfinite(dt)&(dt==0),
       'nonmonotonic_timestamp':np.isfinite(dt)&(dt<0),
       'temporal_gap':np.isfinite(dt)&(dt>cfg.gps_gap_s),
       'abnormal_jump':np.isfinite(speed)&(speed>cfg.gps_jump_mps),
       'stationary':np.isfinite(speed)&(speed<=cfg.stationary_mps),
    }
    df=pd.DataFrame({'partition':partition,'trip_id':trip['trip_id'], 'observation_index':np.arange(n,dtype=np.int64),
      'longitude':lon,'latitude':lat,'timestamp_s':t,'delta_t_s':dt,'distance_from_previous_m':dist,
      'speed_mps':speed,'bearing_deg':bearing,'acceleration_mps2':accel, 'valid_gps':valid&tf})
    for k,v in flags.items(): df[k]=v
    return df

def _projection(nodes):
    valid=valid_lonlat(nodes[:,0],nodes[:,1]);
    if not valid.any(): raise DataError('Road network has no valid coordinate references')
    # UTM metric CRS appropriate for these local networks; unlike degrees, distances are metres.
    lon=float(np.median(nodes[valid,0])); lat=float(np.median(nodes[valid,1]))
    zone=int(np.floor((lon+180)/6)+1); zone=max(1,min(zone,60))
    crs=CRS.from_epsg((32600 if lat>=0 else 32700)+zone)
    return Transformer.from_crs('EPSG:4326',crs,always_xy=True),Transformer.from_crs(crs,'EPSG:4326',always_xy=True),str(crs.to_authority())

def build_road_network(trip):
    nodes=trip['nodes']; arcs=trip['arcs']; n=len(nodes); m=len(arcs)
    good_nodes=valid_lonlat(nodes[:,0],nodes[:,1]); good_endpoints=(arcs>=0).all(axis=1)&(arcs<n).all(axis=1)
    usable=good_endpoints.copy()
    usable[good_endpoints] = good_nodes[arcs[good_endpoints,0]] & good_nodes[arcs[good_endpoints,1]]
    fwd,back,crs_id=_projection(nodes)
    xy=np.full((n,2),np.nan)
    xy[good_nodes,0],xy[good_nodes,1] = fwd.transform(nodes[good_nodes,0],nodes[good_nodes,1])
    start=np.full((m,2),np.nan);end=np.full((m,2),np.nan)
    start[usable] = xy[arcs[usable,0]]; end[usable] = xy[arcs[usable,1]]
    lengths=np.full(m,np.nan)
    lengths[usable] = np.linalg.norm(end[usable]-start[usable],axis=1)
    nonzero=usable & (lengths>1e-7)
    arc_heading=np.full(m,np.nan)
    dx=end[nonzero,0]-start[nonzero,0];dy=end[nonzero,1]-start[nonzero,1]
    arc_heading[nonzero] = np.mod(np.degrees(np.arctan2(dx,dy)),360)
    deg=np.zeros(n,dtype=np.int32)
    np.add.at(deg,arcs[good_endpoints,0],1);np.add.at(deg,arcs[good_endpoints,1],1)
    isolated=int(np.sum(deg==0))
    # Connectivity is weak/undirected for diagnostics; edges remain directed for matching.
    adj=coo_matrix((np.ones(int(nonzero.sum()),dtype=np.int8), (arcs[nonzero,0],arcs[nonzero,1])), shape=(n,n)).tocsr()
    components,ids=connected_components(adj,directed=True,connection='weak')
    counts=np.bincount(ids,minlength=components)
    dup=int(pd.DataFrame(arcs[good_endpoints]).duplicated().sum())
    nodes_df=pd.DataFrame({'node_id':np.arange(n,dtype=np.int64),'longitude':nodes[:,0],'latitude':nodes[:,1],
       'x_m':xy[:,0], 'y_m':xy[:,1], 'valid_node':good_nodes, 'weak_component':ids})
    arcs_df=pd.DataFrame({'arc_id':np.arange(m,dtype=np.int64),'from_node':arcs[:,0], 'to_node':arcs[:,1],
      'length_m':lengths,'heading_deg':arc_heading,'valid_arc':nonzero,
      'zero_length':usable & ~nonzero,'invalid_endpoint':~good_endpoints})
    result={'nodes':nodes_df,'arcs':arcs_df,'coords_xy':xy,'arc_raw':arcs,'start_xy':start,'end_xy':end,
      'lengths':lengths,'heading':arc_heading,'usable':nonzero,'forward':fwd,'backward':back,'crs_id':crs_id,
      'quality':{'node_count':n,'arc_count':m,'invalid_nodes':int((~good_nodes).sum()),
       'invalid_endpoint_arcs':int((~good_endpoints).sum()),'invalid_arcs':int((~nonzero).sum()),
       'duplicate_arcs':dup,'zero_length_arcs':int((usable & ~nonzero).sum()),
       'isolated_nodes':isolated,'weak_connected_components':int(components),
       'largest_weak_component_nodes':int(counts.max() if len(counts) else 0),
       'metric_crs':crs_id,'arc_directionality':'from_node -> to_node; do not assume reverse'}}
    return result

def generate_candidates(observations,road_network,config=None):
    cfg=(config or Config()).validate()
    road=road_network; good_arcs=np.flatnonzero(road['usable'])
    cols=['partition','trip_id','observation_index','candidate_rank','arc_id','distance_m','projection_fraction',
       'snapped_longitude','snapped_latitude','candidate_source','fallback_used']
    if not len(good_arcs): return pd.DataFrame(columns=cols)
    # Shapely bulk construction is substantially faster than Python LineString loops.
    line_xy=np.stack((road['start_xy'][good_arcs],road['end_xy'][good_arcs]),axis=1)
    lines=shapely.linestrings(line_xy)
    tree=STRtree(lines)
    valid=observations['valid_gps'].to_numpy(dtype=bool)
    gps_idx=np.flatnonzero(valid)
    if not len(gps_idx): return pd.DataFrame(columns=cols)
    x,y=road['forward'].transform(observations['longitude'].to_numpy()[gps_idx],observations['latitude'].to_numpy()[gps_idx])
    gps_points=shapely.points(np.stack((x,y),axis=1))
    gps_local, arc_local=tree.query(gps_points, predicate='dwithin',distance=cfg.radius_m)
    fallback_used=np.zeros(len(gps_local),dtype=bool)
    if cfg.fallback:
        present=np.zeros(len(gps_idx),dtype=bool);present[gps_local]=True
        missing=np.flatnonzero(~present)
        if len(missing):
            nearest=tree.nearest(gps_points[missing]); nd=shapely.distance(gps_points[missing],lines[nearest]);
            keep=nd<=cfg.fallback_max_m
            gps_local=np.concatenate((gps_local,missing[keep]))
            arc_local=np.concatenate((arc_local,nearest[keep]))
            fallback_used=np.concatenate((fallback_used,np.ones(int(keep.sum()),dtype=bool)))
    if not len(gps_local):return pd.DataFrame(columns=cols)
    line=lines[arc_local]; p=gps_points[gps_local]
    # Fractional arclength projection and its genuine metric point-to-line distance.
    length=shapely.length(line)
    projected=shapely.line_locate_point(line,p)
    frac=np.clip(projected/length,0.,1.)
    snapped=shapely.line_interpolate_point(line,projected)
    dist=shapely.distance(p,snapped)
    # No geometrically invalid geometries should enter index; reject nonfinite computed distances.
    ok=np.isfinite(dist)&np.isfinite(frac)
    gps_local=gps_local[ok];arc_local=arc_local[ok];dist=dist[ok];frac=frac[ok];snapped=snapped[ok];fallback_used=fallback_used[ok]
    arc_ids=good_arcs[arc_local]
    # Sort by observation, increasing true distance then stable original arc ID.
    order=np.lexsort((arc_ids,dist,gps_local))
    gps_local=gps_local[order];arc_ids=arc_ids[order];dist=dist[order];frac=frac[order];snapped=snapped[order];fallback_used=fallback_used[order]
    # Deduplicate candidates and top-k within each observation, including fallback.
    unique=np.r_[True,(gps_local[1:]!=gps_local[:-1])|(arc_ids[1:]!=arc_ids[:-1])]
    gps_local=gps_local[unique];arc_ids=arc_ids[unique];dist=dist[unique];frac=frac[unique];snapped=snapped[unique];fallback_used=fallback_used[unique]
    first=np.minimum.accumulate(np.where(np.r_[True,gps_local[1:]!=gps_local[:-1]],np.arange(len(gps_local)),np.iinfo(np.int64).max))
    # np.maximum.accumulate is correct for initial position of each group.
    first=np.maximum.accumulate(np.where(np.r_[True,gps_local[1:]!=gps_local[:-1]],np.arange(len(gps_local)),0))
    rank=np.arange(len(gps_local))-first+1
    k=rank<=cfg.top_k
    gps_local=gps_local[k];arc_ids=arc_ids[k];dist=dist[k];frac=frac[k];snapped=snapped[k];fallback_used=fallback_used[k];rank=rank[k]
    slon,slat=road['backward'].transform(shapely.get_x(snapped),shapely.get_y(snapped))
    source=np.where(fallback_used,'nearest_fallback','radius')
    df=pd.DataFrame({'partition':observations['partition'].iloc[0], 'trip_id':observations['trip_id'].iloc[0],
      'observation_index':gps_idx[gps_local], 'candidate_rank':rank.astype(np.int16),'arc_id':arc_ids.astype(np.int64),
      'distance_m':dist, 'projection_fraction':frac,'snapped_longitude':slon,'snapped_latitude':slat,
      'candidate_source':source,'fallback_used':fallback_used})
    return df

def circular_difference_deg(a,b):
    return np.abs((a-b+180.)%360.-180.)

FEATURES=['distance_m','heading_difference_deg','speed_mps','delta_t_s','road_length_m','projection_fraction',
          'acceleration_mps2','gps_previous_step_m','gps_gap','gps_jump','gps_stationary',
          'is_fallback','distance_over_segment_length']

def extract_candidate_features(observations,candidates,road_network):
    """Shared inference/training features: never touches .route, labels, or split membership."""
    road=road_network
    meta_cols=['partition','trip_id','observation_index','arc_id','candidate_rank','distance_m','projection_fraction',
       'snapped_longitude','snapped_latitude','candidate_source','fallback_used']
    if candidates.empty:
        return pd.DataFrame(columns=['partition','trip_id','observation_index','arc_id']+FEATURES)
    c=candidates
    ix=c['observation_index'].to_numpy(dtype=np.int64); aid=c['arc_id'].to_numpy(dtype=np.int64)
    obs=observations.iloc[ix]
    head=obs['bearing_deg'].to_numpy(dtype=float)
    road_heading=road['heading'][aid]
    direction_diff=circular_difference_deg(head,road_heading)
    length=road['lengths'][aid]
    df=pd.DataFrame({'partition':c['partition'].to_numpy(),'trip_id':c['trip_id'].to_numpy(),
      'observation_index':ix,'arc_id':aid,'distance_m':c['distance_m'].to_numpy(dtype=float),
      'heading_difference_deg':direction_diff, 'speed_mps':obs['speed_mps'].to_numpy(dtype=float),
      'delta_t_s':obs['delta_t_s'].to_numpy(dtype=float),'road_length_m':length,
      'projection_fraction':c['projection_fraction'].to_numpy(dtype=float),
      'acceleration_mps2':obs['acceleration_mps2'].to_numpy(dtype=float),
      'gps_previous_step_m':obs['distance_from_previous_m'].to_numpy(dtype=float),
      'gps_gap':obs['temporal_gap'].to_numpy(dtype=np.int8),
      'gps_jump':obs['abnormal_jump'].to_numpy(dtype=np.int8),
      'gps_stationary':obs['stationary'].to_numpy(dtype=np.int8),
      'is_fallback':c['fallback_used'].to_numpy(dtype=np.int8),
      'distance_over_segment_length':np.divide(c['distance_m'].to_numpy(dtype=float),length,
         out=np.full(len(c),np.nan),where=length>0)})
    return df
