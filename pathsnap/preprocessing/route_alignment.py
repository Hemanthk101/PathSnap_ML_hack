"""Reference route integrity and explicitly weak (not gold) geometric alignment.

A .route is an ordered list of arc IDs, NOT one target per track row.
Even high geometric confidence remains a *weak proxy* unless independent
observation-level matching labels are supplied by the organizers.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from .geometry import circular_difference_deg
from .config import Config


def analyze_reference(trip,road,candidates,observations,config=None):
    cfg=(config or Config()).validate()
    route=trip.get('route')
    columns=['partition','trip_id','observation_index','reference_sequence_index','aligned_arc_id',
             'route_distance_m','alignment_margin_m','alignment_heading_difference_deg',
             'alignment_status','alignment_provenance']
    if route is None:
        return pd.DataFrame(columns=columns),pd.DataFrame(columns=['route_position','arc_id','valid_arc','directed_contiguous']),{
            'reference_route_available':False,'weak_alignments':0,'verified_labels':0,'reference_length':0}
    route=np.asarray(route,dtype=np.int64)
    m=len(road['arc_raw']);valid=(route>=0)&(route<m)
    use=np.zeros(len(route),dtype=bool)
    use[valid]=road['usable'][route[valid]]
    contiguous=np.zeros(len(route),dtype=bool)
    if len(route)>1:
        curr=route[1:];prev=route[:-1];ok=valid[1:]&valid[:-1]
        contiguous[1:][ok]=road['arc_raw'][prev[ok],1]==road['arc_raw'][curr[ok],0]
    route_df=pd.DataFrame({'route_position':np.arange(len(route)), 'arc_id':route,
                           'valid_arc':use,'directed_contiguous':contiguous})
    quality={'reference_route_available':True,'reference_length':len(route),'invalid_route_arc_ids':int((~valid).sum()),
             'unusable_route_arcs':int((~use).sum()),'broken_route_connections':int((~contiguous[1:]).sum()),
             'duplicate_route_arc_entries':int(len(route)-len(np.unique(route))),
             'verified_labels':0,'alignment_type':'weak_geometry_inference_not_ground_truth'}
    if not use.any():
        quality['weak_alignments']=0
        return pd.DataFrame(columns=columns),route_df,quality
    # Candidate alignment is independently computed against route geometry, not candidate-ranked nearest roads.
    seq=np.flatnonzero(use); a=route[seq]
    lines=shapely.linestrings(np.stack((road['start_xy'][a],road['end_xy'][a]),axis=1))
    tree=STRtree(lines)
    valid_obs=np.flatnonzero(observations['valid_gps'].to_numpy(dtype=bool))
    if not len(valid_obs):
        quality['weak_alignments']=0
        return pd.DataFrame(columns=columns),route_df,quality
    lon=observations['longitude'].to_numpy()[valid_obs];lat=observations['latitude'].to_numpy()[valid_obs]
    x,y=road['forward'].transform(lon,lat);points=shapely.points(np.stack((x,y),axis=1))
    # Radius includes competing nearby reference segments, so ambiguity can be measured.
    gps_idx,seg_idx=tree.query(points,predicate='dwithin',distance=max(60.,cfg.weak_label_distance_m+cfg.weak_label_margin_m))
    if not len(gps_idx):
        quality['weak_alignments']=0
        return pd.DataFrame(columns=columns),route_df,quality
    d=shapely.distance(points[gps_idx],lines[seg_idx]);head=observations['bearing_deg'].to_numpy()[valid_obs[gps_idx]]
    diff=circular_difference_deg(head,road['heading'][a[seg_idx]])
    # Penalize opposite-direction matches where heading is defined; fallback to spatial only for very slow points,
    # but never assign weak targets without valid moving heading.
    order=np.lexsort((seg_idx,d,gps_idx))
    gps_idx,seg_idx,d,diff=gps_idx[order],seg_idx[order],d[order],diff[order]
    first=np.r_[True,gps_idx[1:]!=gps_idx[:-1]]
    best=np.flatnonzero(first)
    second_d=np.full(len(best),np.inf)
    starts=np.r_[best,len(d)]
    # Only next same-observation alternative with a different arc matters for geometric ambiguity.
    for j,k in enumerate(best):
        q=k+1
        while q<starts[j+1] and a[seg_idx[q]]==a[seg_idx[k]]:
            q+=1
        if q<starts[j+1]:second_d[j]=d[q]
    best_gps=valid_obs[gps_idx[best]];best_seq=seq[seg_idx[best]]
    best_arc=a[seg_idx[best]];best_dist=d[best];best_diff=diff[best]
    margin=second_d-best_dist
    candidate_map={int(i):set(map(int,part['arc_id'].to_numpy())) for i,part in candidates.groupby('observation_index',sort=False)}
    # Conservative forward consistency: candidate sequence positions must be nondecreasing across accepted observations.
    preliminary=(best_dist<=cfg.weak_label_distance_m)&(margin>=cfg.weak_label_margin_m)&np.isfinite(best_diff)&(best_diff<=cfg.weak_label_heading_deg)
    # Neighbor support: isolated single-point guesses in a noisy trajectory are unreliable.
    supported=np.zeros(len(best),dtype=bool)
    for j in np.flatnonzero(preliminary):
        neighbor=[]
        if j>0 and preliminary[j-1] and best_gps[j]-best_gps[j-1]<=5:neighbor.append(j-1)
        if j+1<len(best) and preliminary[j+1] and best_gps[j+1]-best_gps[j]<=5:neighbor.append(j+1)
        supported[j]=any((abs(int(best_seq[j])-int(best_seq[k]))<=10) for k in neighbor)
    allowed=preliminary&supported
    # Reject proposed non-monotone labels (do not force alignment via incorrect reassignment).
    last_seq=-1
    for j in range(len(best)):
        if not allowed[j]:continue
        if best_seq[j] < last_seq-cfg.weak_label_max_route_regression:
            allowed[j]=False
        else: last_seq=int(best_seq[j])
    present=np.array([int(i) in candidate_map and int(b) in candidate_map[int(i)] for i,b in zip(best_gps,best_arc)])
    # Weak labels only if the positive arc exists in the candidate set.
    alignment_status=np.where(allowed & present, 'weak_supported',np.where(allowed,'weak_candidate_miss','ambiguous_unlabeled'))
    df=pd.DataFrame({'partition':observations['partition'].iloc[0], 'trip_id':trip['trip_id'],
        'observation_index':best_gps,'reference_sequence_index':best_seq,'aligned_arc_id':best_arc,
        'route_distance_m':best_dist,'alignment_margin_m':np.where(np.isfinite(margin),margin,np.nan),
        'alignment_heading_difference_deg':best_diff,'alignment_status':alignment_status,
        'alignment_provenance':'geometric_heuristic_not_verified_ground_truth'})
    quality.update({'weak_alignments':int((alignment_status=='weak_supported').sum()),
      'weak_candidate_misses':int((alignment_status=='weak_candidate_miss').sum()),
      'ambiguous_observations_evaluated':int((alignment_status=='ambiguous_unlabeled').sum()),
      'observations_with_no_nearby_route_geometry':int(len(observations)-len(best)),
      'observations_with_any_route_geometry':int(len(best))})
    return df,route_df,quality


def derive_weak_targets(candidates,alignment):
    """Candidate-level proxy targets, positive only when geometrically supported.

    Unsafe unaligned candidates do not get negative labels. No gold labels are claimed.
    """
    cols=['partition','trip_id','observation_index','arc_id','weak_target','label_status','label_provenance']
    if candidates.empty or alignment.empty:return pd.DataFrame(columns=cols)
    a=alignment.loc[alignment['alignment_status']=='weak_supported',
      ['partition','trip_id','observation_index','aligned_arc_id']].copy()
    if a.empty:return pd.DataFrame(columns=cols)
    c=candidates[['partition','trip_id','observation_index','arc_id']]
    joined=c.merge(a,how='inner',on=['partition','trip_id','observation_index'],validate='many_to_one')
    joined['weak_target']=(joined['arc_id']==joined['aligned_arc_id']).astype(np.int8)
    joined['label_status']='weak_not_verified'
    joined['label_provenance']='route_geometry_spatial_margin_heading_monotonicity'
    out=joined[cols]
    assert (out.groupby(['partition','trip_id','observation_index'])['weak_target'].sum()==1).all()
    return out
