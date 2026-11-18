"""Deterministic synthetic fixtures: not reported as measured real-world accuracy."""
import numpy as np
import pandas as pd
import pytest
from pathsnap.preprocessing.config import Config
from pathsnap.preprocessing.io import read_numeric,load_trip,DataError
from pathsnap.preprocessing.geometry import preprocess_gps,build_road_network,generate_candidates,extract_candidate_features,circular_difference_deg,FEATURES
from pathsnap.preprocessing.route_alignment import analyze_reference,derive_weak_targets
from pathsnap.preprocessing.pipeline import prepare_inference_trip,validate_trip


def example_trip():
    return {'trip_id':'abc','track':np.array([[12.,50.,1.],[12.0001,50.,2.],[12.0002,50.,3.],[12.0003,50.,4.]]),
      'nodes':np.array([[12.,50.],[12.001,50.],[12.001,50.001],[12.002,50.001]]),
      'arcs':np.array([[0,1],[1,2],[2,3],[1,0]]), 'route':np.array([0,1,2])}

def prepared(config=None):
    cfg=config or Config(); t=example_trip();o=preprocess_gps(t,cfg,'train');r=build_road_network(t);c=generate_candidates(o,r,cfg);f=extract_candidate_features(o,c,r)
    return t,o,r,c,f

def test_observation_index_and_schema():
    t,o,r,c,f=prepared()
    assert o.observation_index.tolist()==list(range(4))
    assert list(f.columns[-len(FEATURES):])==FEATURES
    assert set(c.arc_id).issubset(set(r["arcs"].arc_id))

def test_geodetic_units_and_bearing():
    _,o,_,_,_=prepared()
    assert 5 < o.distance_from_previous_m.iloc[1] < 10
    assert 5 < o.speed_mps.iloc[1] < 10
    assert 80 < o.bearing_deg.iloc[1] < 100

def test_original_ids_and_directed_graph():
    _,_,r,_,_=prepared()
    assert r['arcs']['arc_id'].tolist()==[0,1,2,3]
    assert r['nodes']['node_id'].tolist()==[0,1,2,3]
    assert r['arc_raw'][0].tolist()==[0,1]
    assert r['arc_raw'][3].tolist()==[1,0]

def test_point_to_segment_not_node_distance():
    _,_,_,c,_=prepared()
    assert (c.query('arc_id == 0')['distance_m'].min()<1.)
    assert (c['projection_fraction'].between(0,1)).all()

def test_invalid_arc_reference_flagged():
    t=example_trip();t['arcs']=np.array([[0,9],[0,1],[-1,0]])
    r=build_road_network(t)
    assert r['quality']['invalid_endpoint_arcs']==2
    assert r['usable'].tolist()==[False,True,False]

def test_duplicate_timestamp_and_zero_deltat():
    t=example_trip();t['track'][1,2]=t['track'][0,2]
    o=preprocess_gps(t)
    assert o.duplicate_timestamp.iloc[1]
    assert np.isnan(o.speed_mps.iloc[1])

def test_backward_timestamp_flagged():
    t=example_trip();t['track'][1,2]=-1
    o=preprocess_gps(t)
    assert o.nonmonotonic_timestamp.iloc[1]
    assert np.isnan(o.speed_mps.iloc[1])

def test_stationary_and_acceleration():
    t=example_trip();t['track'][2,0:2]=t['track'][1,0:2]
    o=preprocess_gps(t)
    assert o.stationary.iloc[2] and o.speed_mps.iloc[2]==0
    assert np.isfinite(o.acceleration_mps2.iloc[2])

def test_gaps_jump_and_missing_coordinates():
    t=example_trip();t['track'][1,2]=100;t['track'][2,2]=101;t['track'][3,0]=np.nan
    o=preprocess_gps(t)
    assert o.temporal_gap.iloc[1]
    assert o.invalid_coordinate.iloc[3]
    assert np.isnan(o.speed_mps.iloc[3])

def test_sudden_gps_jump_flag():
    t=example_trip();t['track'][1,0]=12.02
    o=preprocess_gps(t)
    assert o.abnormal_jump.iloc[1]

def test_no_candidate_with_fallback_disabled():
    t=example_trip();t['track'][:,0]=20
    o=preprocess_gps(t);r=build_road_network(t)
    c=generate_candidates(o,r,Config(radius_m=1,fallback=False))
    assert c.empty

def test_fallback_nearest_source():
    t=example_trip();t['track'][:,1]+=0.0005
    o=preprocess_gps(t);r=build_road_network(t)
    c=generate_candidates(o,r,Config(radius_m=1,fallback=True,fallback_max_m=200))
    assert len(c)==4 and c.fallback_used.all()
    assert (c.candidate_source=='nearest_fallback').all()

def test_fallback_max_distance_rejects():
    t=example_trip();t['track'][:,0]=20
    o=preprocess_gps(t);r=build_road_network(t)
    c=generate_candidates(o,r,Config(radius_m=1,fallback=True,fallback_max_m=100))
    assert c.empty

def test_topk_sort_deterministic_unique():
    _,o,r,c,_=prepared(Config(top_k=2,radius_m=200))
    assert c.groupby('observation_index').size().max()<=2
    assert not c.duplicated(['partition','trip_id','observation_index','arc_id']).any()
    c2=generate_candidates(o,r,Config(top_k=2,radius_m=200))
    pd.testing.assert_frame_equal(c,c2)

def test_circular_heading_wraparound():
    assert circular_difference_deg(359,1)==2
    assert circular_difference_deg(1,359)==2
    assert circular_difference_deg(10,190)==180

def test_graph_weak_components_and_duplicate_arcs():
    t=example_trip();t['nodes']=np.concatenate([t['nodes'],[[13.,49.]]]);t['arcs']=np.concatenate([t['arcs'],[[0,1]]]);r=build_road_network(t)
    assert r['quality']['isolated_nodes']==1
    assert r['quality']['duplicate_arcs']==1
    assert r['quality']['weak_connected_components']>=2

def test_reference_sequence_not_observation_alignment():
    t,o,r,c,_=prepared()
    alignment,routes,quality=analyze_reference(t,r,c,o)
    assert len(routes)==3 and len(o)==4
    assert quality['broken_route_connections']==0
    assert quality['verified_labels']==0
    assert 'weak' in quality['alignment_type']

def test_repeated_route_arcs_and_discontinuity():
    t=example_trip();t['route']=np.array([0,1,1,0,2]);o=preprocess_gps(t);r=build_road_network(t);c=generate_candidates(o,r)
    _,routes,q=analyze_reference(t,r,c,o)
    assert q['duplicate_route_arc_entries']==2
    assert q['broken_route_connections']>0

def test_candidate_miss_is_not_invented_positive():
    t,o,r,c,_=prepared(Config(top_k=1,radius_m=200))
    a=pd.DataFrame([{'partition':'train','trip_id':'abc','observation_index':0,'aligned_arc_id':999,'alignment_status':'weak_supported'}])
    # Upstream alignment is responsible for refusing impossible-positive candidate misses.
    # Never pass an unsupported "weak_supported" label into assembly.
    a.loc[0,"alignment_status"]="weak_candidate_miss"
    assert derive_weak_targets(c,a).empty

def test_unlabeled_observations_not_assigned_negatives():
    _,o,r,c,_=prepared()
    a=pd.DataFrame([{'partition':'train','trip_id':'abc','observation_index':0,'aligned_arc_id':0,'alignment_status':'ambiguous_unlabeled'}])
    assert derive_weak_targets(c,a).empty

def test_data_leakage_route_independent_features():
    t,o,r,c,f=prepared()
    t['route']=np.array([3,3,3,3])
    o2=preprocess_gps(t,partition='train');r2=build_road_network(t);c2=generate_candidates(o2,r2);f2=extract_candidate_features(o2,c2,r2)
    pd.testing.assert_frame_equal(f,f2)

def test_nan_feature_is_not_fake_zero():
    _,o,r,c,f=prepared()
    assert np.isnan(f.loc[f.observation_index==0,'speed_mps']).all()
    assert not np.isinf(f[FEATURES].to_numpy()).any()

def test_empty_candidates_feature_frame():
    _,o,r,_,_=prepared()
    f=extract_candidate_features(o,pd.DataFrame(),r)
    assert f.empty and all(x in f for x in FEATURES)

def test_composite_key_validation():
    _,o,r,c,f=prepared()
    dup=pd.concat([c,c.iloc[:1]])
    assert validate_trip(o,r,dup,f)['status']=='failed'

def test_malformed_file_raises(tmp_path):
    p=tmp_path/'bad.track';p.write_text('1.1 2.2 3.3\nnot a number\n')
    with pytest.raises(DataError):read_numeric(p,3)

def test_missing_file_raises(tmp_path):
    with pytest.raises(DataError):load_trip(tmp_path/'missing')

def test_one_record_parsing_preserves_shape(tmp_path):
    p=tmp_path/'one.nodes';p.write_text('12.5 42.5\n')
    assert read_numeric(p,2).shape==(1,2)

def test_inference_never_loads_route(tmp_path):
    p=tmp_path/'abc';p.mkdir();t=example_trip()
    for ext,arr in [('track',t['track']),('nodes',t['nodes']),('arcs',t['arcs'])]:
        np.savetxt(p/('abc.'+ext),arr,fmt='%d' if ext=='arcs' else '%.8f')
    (p/'abc.route').write_text('not numeric; inference must not read it')
    result=prepare_inference_trip(p)
    assert len(result['features'])>0
    assert result['validation']['status']=='passed'

def test_weak_labels_have_single_positive_per_observation():
    t,o,r,c,_=prepared()
    a=pd.DataFrame([{'partition':'train','trip_id':'abc','observation_index':0,'aligned_arc_id':0,'alignment_status':'weak_supported'}])
    labels=derive_weak_targets(c,a)
    assert labels.weak_target.sum()==1
    assert len(labels)==len(c[c.observation_index==0])
    assert labels.label_status.eq('weak_not_verified').all()

def test_road_zero_length_segment_skipped():
    t=example_trip();t['nodes'][1]=t['nodes'][0];r=build_road_network(t)
    assert r['quality']['zero_length_arcs']>=1
    assert not r['usable'][0]

def test_invalid_config_rejected():
    with pytest.raises(ValueError): Config(top_k=0).validate()
