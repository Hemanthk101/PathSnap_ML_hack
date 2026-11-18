"""End-to-end discovery, per-trip processing, validation and reusable inference."""
import csv,gzip,json,time,os,logging,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import resource
except ImportError:
    resource = None
from .config import Config,SCHEMA_VERSION
from .io import discover,load_trip,write_table,read_table,write_json,DataError,HAS_ARROW
from .geometry import preprocess_gps,build_road_network,generate_candidates,extract_candidate_features,FEATURES
from .route_alignment import analyze_reference,derive_weak_targets

LOG=logging.getLogger('pathsnap')
BASE_KEYS=['partition','trip_id','observation_index','arc_id']

def _append_table(df,base):
    """Incrementally write cross-trip CSV.gz regardless of optional Parquet availability."""
    p=Path(str(base)+'.csv.gz');p.parent.mkdir(parents=True,exist_ok=True)
    header=not p.exists()
    with gzip.open(p,'at',encoding='utf8',newline='',compresslevel=1) as f:
        df.to_csv(f,index=False,header=header,float_format='%.12g')
    return str(p)

def _write_graph(road,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, from_node=road['arc_raw'][:,0],to_node=road['arc_raw'][:,1],
                        arc_id=np.arange(len(road['arc_raw']),dtype=np.int64),
                        valid_arc=road['usable'],length_m=road['lengths'])

def feature_schema():
    info={'distance_m':('m','Actual Euclidean shortest distance to projected segment'),
      'heading_difference_deg':('degrees','Circular absolute difference between GPS bearing and directed arc heading'),
      'speed_mps':('m/s','Geodesic inter-observation speed (undefined for nonpositive dt)'),
      'delta_t_s':('s','Elapsed time since previous original observation'),
      'road_length_m':('m','Projected metric segment length'),
      'projection_fraction':('unitless','0-1 fraction along from_node -> to_node'),
      'acceleration_mps2':('m/s^2','Difference of two defined consecutive speeds per elapsed second'),
      'gps_previous_step_m':('m','WGS84 geodesic distance from previous valid observation'),
      'gps_gap':('boolean 0/1','Whether elapsed time exceeds configured gap threshold'),
      'gps_jump':('boolean 0/1','Whether observed geodesic speed exceeds configured jump threshold'),
      'gps_stationary':('boolean 0/1','Whether finite observed speed is below stationary threshold'),
      'is_fallback':('boolean 0/1','Nearest-road fallback because no radius candidate existed'),
      'distance_over_segment_length':('unitless','Actual candidate distance divided by road length')}
    return {'schema_version':SCHEMA_VERSION,'feature_order':FEATURES,
      'features':[{'name':k,'type':'float64' if k not in ('gps_gap','gps_jump','gps_stationary','is_fallback') else 'int8',
        'units':info[k][0],'meaning':info[k][1],
        'missing_value':'NaN means not mathematically defined; downstream train-only imputation required',
        'train_available':True,'inference_available':True} for k in FEATURES],
      'identifiers':BASE_KEYS,'targets_not_features':True,
      'standardization':'Fit all imputers/scalers on training folds ONLY, grouped by trip',
      'labels':'When produced, weak_target is heuristically aligned, NOT gold ground truth'}

def validate_trip(observations,road,candidates,features,weak_targets=None):
    issues=[]; m=len(observations)
    if not np.array_equal(observations['observation_index'].to_numpy(),np.arange(m)):
        issues.append('Original GPS observation indexing lost')
    if candidates.duplicated(BASE_KEYS).any():issues.append('Duplicate composite candidate keys')
    if features.duplicated(BASE_KEYS).any():issues.append('Duplicate feature composite keys')
    if len(features)!=len(candidates):issues.append('Candidate/features count mismatch')
    if candidates['observation_index'].ge(m).any() or candidates['observation_index'].lt(0).any():
        issues.append('Candidate references nonexistent GPS observation')
    if candidates['arc_id'].ge(len(road['arcs'])).any() or candidates['arc_id'].lt(0).any():
        issues.append('Candidate references nonexistent arc')
    if not candidates.empty and not road['usable'][candidates['arc_id'].to_numpy()].all():
        issues.append('Candidate references invalid arc')
    if len(candidates):
        if ((candidates['projection_fraction']<0)|(candidates['projection_fraction']>1)).any():
            issues.append('Projection fraction outside 0..1')
        if not np.isfinite(candidates['distance_m']).all() or candidates['distance_m'].lt(0).any():
            issues.append('Invalid point-to-line distance')
        if not (candidates['distance_m'].to_numpy()>=0).all():issues.append('Negative distance')
        groups=candidates.groupby('observation_index',sort=False)
        if groups.size().max()>65535:issues.append('Implausibly large candidate group')
        if not candidates['candidate_rank'].eq(groups.cumcount()+1).all():
            issues.append('Candidate ranks not sequential')
        if groups['distance_m'].diff().lt(-1e-7).any():
            issues.append('Candidates not ranked by true distance')
    for name in FEATURES:
        if name not in features:issues.append('Missing feature '+name)
        elif np.isinf(features[name].to_numpy(dtype=float)).any():issues.append('Infinite feature '+name)
    if weak_targets is not None and not weak_targets.empty:
        if weak_targets.duplicated(BASE_KEYS).any():issues.append('Duplicate weak targets')
        sums=weak_targets.groupby(['partition','trip_id','observation_index'])['weak_target'].sum()
        if not (sums==1).all():issues.append('Weak target observation lacks exactly one positive')
        if not set(map(tuple,weak_targets[BASE_KEYS].to_numpy())).issubset(set(map(tuple,candidates[BASE_KEYS].to_numpy()))):
            issues.append('Weak targets reference nonexistent candidates')
    report={'status':'passed' if not issues else 'failed','failures':issues,
      'gps_observations':m,'valid_gps_observations':int(observations['valid_gps'].sum()),
      'observations_with_candidates':int(candidates['observation_index'].nunique()),
      'observations_without_candidates':int(m-candidates['observation_index'].nunique()),
      'candidate_rows':len(candidates),'weak_target_candidate_rows':len(weak_targets) if weak_targets is not None else 0,
      'verified_ground_truth_rows':0,'feature_rows':len(features),
      'feature_columns':FEATURES,'feature_nan_counts':{k:int(features[k].isna().sum()) for k in FEATURES if k in features}}
    return report

def process_one_trip(trip_path,partition='train',out_dir=None,config=None, use_reference=True):
    cfg=(config or Config()).validate()
    t0=time.perf_counter()
    trip=load_trip(trip_path, load_route=use_reference)
    obs=preprocess_gps(trip,cfg,partition=partition)
    road=build_road_network(trip)
    t_candidates=time.perf_counter()
    candidates=generate_candidates(obs,road,cfg)
    candidate_s=time.perf_counter()-t_candidates
    features=extract_candidate_features(obs,candidates,road)
    if use_reference:
        align,routes,label_quality=analyze_reference(trip,road,candidates,obs,cfg)
        weak_targets=derive_weak_targets(candidates,align)
    else:
        align=None;routes=None;weak_targets=None
        label_quality={'reference_route_available':False,'verified_labels':0,'weak_alignments':0,
                       'note':'Reference data intentionally excluded for leakage-safe inference or student evaluation'}
    validation=validate_trip(obs,road,candidates,features,weak_targets)
    if validation['status']!='passed':
        raise DataError(f'{partition}/{trip["trip_id"]} validation failed: {validation["failures"]}')
    elapsed=time.perf_counter()-t0
    source=Path(trip_path)
    config_sha256=hashlib.sha256(json.dumps(cfg.as_json(),sort_keys=True).encode()).hexdigest()
    source_signature={ext:[(source/f"{trip['trip_id']}.{ext}").stat().st_size,(source/f"{trip['trip_id']}.{ext}").stat().st_mtime_ns] for ext in (('track','nodes','arcs','route') if use_reference else ('track','nodes','arcs'))}
    manifest={'config_sha256':config_sha256,'input_signature':source_signature,'partition':partition,'trip_id':trip['trip_id'],
      'observations':len(obs),'nodes':len(road['nodes']),'arcs':len(road['arcs']),
      'reference_route_entries':len(trip['route']) if trip['route'] is not None else None,
      'invalid_observations':int((~obs['valid_gps']).sum()),
      'missing_or_invalid_timestamp':int(obs['invalid_timestamp'].sum()),
      'duplicate_timestamp':int(obs['duplicate_timestamp'].sum()),
      'nonmonotonic_timestamp':int(obs['nonmonotonic_timestamp'].sum()),
      'temporal_gaps':int(obs['temporal_gap'].sum()),'abnormal_jumps':int(obs['abnormal_jump'].sum()),
      'candidate_rows':len(candidates),'candidate_observations':int(candidates['observation_index'].nunique()),
      'missing_candidate_observations':int(len(obs)-candidates['observation_index'].nunique()),
      'fallback_candidate_rows':int(candidates['fallback_used'].sum()) if len(candidates) else 0,
      'weak_labeled_observations':int(weak_targets['observation_index'].nunique()) if weak_targets is not None else 0,
      'processing_seconds':elapsed,'candidate_generation_seconds':candidate_s,
      'validation_status':validation['status'],'invalid_road_arcs':road['quality']['invalid_arcs'],
      'broken_reference_route_connections':label_quality.get('broken_route_connections')}
    if out_dir is not None:
        out=Path(out_dir)/partition/trip['trip_id'];out.mkdir(parents=True,exist_ok=True)
        write_table(obs,out/'gps_observations')
        write_table(road['nodes'],out/'road_nodes')
        write_table(road['arcs'],out/'road_arcs')
        write_table(candidates,out/'road_candidates')
        write_table(features,out/'candidate_features')
        _write_graph(road,out/'road_graph.npz')
        if routes is not None:
            write_table(routes,out/'reference_routes')
            write_table(align,out/'observation_weak_alignment')
            write_table(weak_targets,out/'weak_targets')
        write_json(road['quality'],out/'road_network_quality.json')
        write_json(label_quality,out/'label_quality_report.json')
        write_json(validation,out/'dataset_validation_report.json')
        write_json(manifest,out/'trip_manifest.json')
    LOG.info('%s/%s obs=%d arcs=%d candidates=%d coverage=%.3f weak=%d %.2fs',partition,trip['trip_id'],len(obs),len(road['arcs']),len(candidates),manifest['candidate_observations']/len(obs),manifest['weak_labeled_observations'],elapsed)
    return {'observations':obs,'road':road,'candidates':candidates,'features':features,
            'alignment':align,'weak_targets':weak_targets,'routes':routes,'label_quality':label_quality,
            'manifest':manifest,'validation':validation}

def prepare_inference_trip(trip_path,config=None):
    """One documented API entrypoint; WILL NOT OPEN .route even when present."""
    res=process_one_trip(trip_path,partition='inference',out_dir=None,config=config,use_reference=False)
    return {k:res[k] for k in ('observations','candidates','features','road','validation')}

def prepare_training_dataset(data_root,out_dir,config=None, partition=None,allow_student_references=False, stop_after=None):
    """Incremental per-trip storage; aggregate feature/metadata/weak-target CSVs built in one pass."""
    cfg=(config or Config()).validate();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    parts=[partition] if partition else ['train','student_test']
    for part in parts:
        for name in (f'{part}_features.csv.gz',f'{part}_candidate_metadata.csv.gz',f'{part}_targets.csv.gz'):
            (out/name).unlink(missing_ok=True)
    write_json(cfg.as_json(),out/'preprocessing_config.json')
    write_json(feature_schema(),out/'feature_schema.json')
    records=[];errors=[];road_quality=[];label_quality=[];validation_records=[]
    entries=discover(data_root,partition=partition)
    if stop_after is not None: entries=entries[:stop_after]
    start=time.perf_counter()
    for part,tid,path in entries:
        try:
            permit_ref=part=='train' or (part=='student_test' and allow_student_references)
            trip_out=out/part/tid
            manifest_path=trip_out/'trip_manifest.json'
            # Resume safely from completed per-trip outputs, but not partially written trips.
            required=[trip_out/'dataset_validation_report.json',trip_out/'road_network_quality.json',
                      trip_out/'label_quality_report.json']
            if manifest_path.exists() and all(x.exists() for x in required) and \
               manifest_path.stat().st_mtime >= max((path/f'{tid}.{e}').stat().st_mtime for e in ('track','nodes','arcs')):
                try:
                    manifest=json.loads(manifest_path.read_text())
                    source_sig={ext:[(path/f'{tid}.{ext}').stat().st_size,(path/f'{tid}.{ext}').stat().st_mtime_ns] for ext in (('track','nodes','arcs','route') if permit_ref else ('track','nodes','arcs'))}
                    config_hash=hashlib.sha256(json.dumps(cfg.as_json(),sort_keys=True).encode()).hexdigest()
                    if manifest.get('config_sha256')!=config_hash or manifest.get('input_signature')!=source_sig or manifest['validation_status']!='passed' or (permit_ref and not (trip_out/'weak_targets.csv.gz').exists() and not (trip_out/'weak_targets.parquet').exists()):
                        raise ValueError('Incomplete cached trip')
                    features=read_table(trip_out/'candidate_features')
                    candidates=read_table(trip_out/'road_candidates')
                    wt=read_table(trip_out/'weak_targets') if permit_ref else None
                    road_q=json.loads((trip_out/'road_network_quality.json').read_text())
                    label_q=json.loads((trip_out/'label_quality_report.json').read_text())
                    val=json.loads((trip_out/'dataset_validation_report.json').read_text())
                    LOG.info('REUSED validated trip %s/%s',part,tid)
                except Exception:
                    manifest=None
            else:
                manifest=None
            if manifest is None:
                result=process_one_trip(path,part,out,cfg,use_reference=permit_ref)
                manifest=result['manifest'];features=result['features'];candidates=result['candidates']
                wt=result['weak_targets'];road_q=result['road']['quality'];label_q=result['label_quality'];val=result['validation']
            records.append(manifest);road_quality.append({'partition':part,'trip_id':tid,**road_q})
            label_quality.append({'partition':part,'trip_id':tid,**label_q})
            validation_records.append({'partition':part,'trip_id':tid,**val})
            _append_table(features[BASE_KEYS+FEATURES],out/f'{part}_features')
            _append_table(candidates[['partition','trip_id','observation_index','arc_id','candidate_rank','snapped_longitude','snapped_latitude','candidate_source','fallback_used']],out/f'{part}_candidate_metadata')
            if wt is not None:
                _append_table(wt,out/f'{part}_targets')
        except Exception as e:
            LOG.exception('FAILED %s/%s',part,tid)
            errors.append({'partition':part,'trip_id':tid,'error_type':type(e).__name__,'message':str(e)})
    total=time.perf_counter()-start
    manifest=pd.DataFrame(records)
    if len(manifest):
        manifest.to_csv(out/'dataset_manifest.csv',index=False)
        manifest.to_csv(out/'eda_summary.csv',index=False)
    write_json({'schema_version':SCHEMA_VERSION,'partitions_requested':parts,'trips_discovered':len(entries),
       'trips_processed':len(records),'errors':errors,'observations':int(manifest['observations'].sum()) if len(manifest) else 0,
       'nodes_across_trip_networks':int(manifest['nodes'].sum()) if len(manifest) else 0,
       'arcs_across_trip_networks':int(manifest['arcs'].sum()) if len(manifest) else 0,
       'candidate_rows':int(manifest['candidate_rows'].sum()) if len(manifest) else 0,
       'storage_format':'parquet' if HAS_ARROW else 'gzip_csv_fallback_missing_pyarrow',
       'student_test_reference_policy':'excluded by default' if not allow_student_references else 'allowed by explicit flag'},out/'dataset_inventory.json')
    write_json({'status':'passed' if not errors else 'partial_with_failures','per_trip':validation_records,
      'failed_trips':errors,'feature_order':FEATURES,'verified_gold_labels':0,
      'label_caution':'Training targets are weak geospatial proxies. They are NOT organizer-provided GPS-to-arc gold labels.'},out/'dataset_validation_report.json')
    write_json({'per_trip':road_quality},out/'road_network_quality.json')
    write_json({'per_trip':label_quality,'verified_gold_labels':0},out/'label_quality_report.json')
    write_json({'per_trip':[{'partition':r['partition'],'trip_id':r['trip_id'],
       'candidate_observations':r['candidate_observations'], 'observations':r['observations'],
       'coverage_fraction':r['candidate_observations']/r['observations'] if r['observations'] else None,
       'weak_labeled_observations':r['weak_labeled_observations']} for r in records]},out/'candidate_coverage_report.json')
    write_json({'total_processing_seconds':total,'trips_completed':len(records),
      'sum_candidate_generation_seconds':float(manifest['candidate_generation_seconds'].sum()) if len(manifest) else 0,
      'per_trip_seconds':{r['partition']+'/'+r['trip_id']:r['processing_seconds'] for r in records},
      'peak_process_rss_kib_os_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None},out/'benchmark_report.json')
    return {'trips_discovered':len(entries),'trips_processed':len(records),'errors':errors,'seconds':total,'records':records}
