"""Dataset-wide input accounting, feature/target key audits, and evidence reports."""
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from .io import read_table,write_json,discover
from .geometry import FEATURES
from .pipeline import prepare_inference_trip

KEYS=['partition','trip_id','observation_index','arc_id']

def audit_processed(data_root,out_dir):
    out=Path(out_dir);m=pd.read_csv(out/'dataset_manifest.csv',dtype={'trip_id':str})
    errors=[];nkeys={};nweak={}
    all_groups=set()
    for part in ('train','student_test'):
        features=read_table(out/f'{part}_features')
        candidates=read_table(out/f'{part}_candidate_metadata')
        expected=int(m.loc[m.partition==part,'candidate_rows'].sum())
        if len(features)!=len(candidates) or len(features)!=expected:
            errors.append(f'{part}: candidate/features count differs from manifest')
        if features[KEYS].duplicated().any() or candidates[KEYS].duplicated().any():
            errors.append(f'{part}: duplicate composite candidate key')
        if not features[KEYS].equals(candidates[KEYS]):
            errors.append(f'{part}: feature/metadata composite keys are not aligned')
        if np.isinf(features[FEATURES].to_numpy(dtype=float)).any():
            errors.append(f'{part}: infinity found in a model feature')
        if set(features.trip_id.astype(str))!=set(m.loc[m.partition==part,'trip_id']):
            errors.append(f'{part}: missing trip IDs in aggregate features')
        nkeys[part]=len(features)
        these=set(features.trip_id.astype(str))
        if all_groups.intersection(these):errors.append('train/student_test trip overlap')
        all_groups.update(these)
        if part=='train':
            targets=read_table(out/'train_targets')
            if targets[KEYS].duplicated().any():errors.append('train: duplicate weak target keys')
            sums=targets.groupby(['trip_id','observation_index'])['weak_target'].agg(['sum','size'])
            if not sums['sum'].eq(1).all():errors.append('Weak-labeled observation has zero or multiple positives')
            check=targets[KEYS].merge(features[KEYS],how='left',on=KEYS,indicator=True,validate='one_to_one')
            if not check._merge.eq('both').all():errors.append('Train targets reference absent candidates')
            if not targets.label_status.eq('weak_not_verified').all():errors.append('Gold labeling incorrectly claimed')
            nweak[part]=int(sums.shape[0])
        else:
            if (out/'student_test_targets.csv.gz').exists():errors.append('Potential student-test target leakage')
            if list((out/'student_test').glob('*/weak_targets.*')):
                errors.append('Potential trip-level student-test weak target leakage')
            nweak[part]=0
    # Re-run one actual training trip as inference, no route input used.
    first=sorted((Path(data_root)/'train').glob('*'))
    if first:
        t=first[0]
        infer=prepare_inference_trip(t)
        trained=read_table(out/'train'/t.name/'candidate_features')
        if not np.array_equal(infer['features'][['observation_index','arc_id']].to_numpy(),trained[['observation_index','arc_id']].to_numpy()):
            errors.append('Real-trip training/inference candidate identities differ')
        elif not np.allclose(infer['features'][FEATURES].to_numpy(dtype=float),trained[FEATURES].to_numpy(dtype=float),equal_nan=True,rtol=1e-8,atol=1e-7):
            errors.append('Real-trip training/inference numerical features differ')
    from json import loads
    roadq=loads((out/'road_network_quality.json').read_text())['per_trip']
    labq=loads((out/'label_quality_report.json').read_text())['per_trip']
    geodata={'data_source':'real supplied archive, not synthetic fixtures',
      'total_trips':len(m),'training_trips':int((m.partition=='train').sum()),
      'student_test_trips':int((m.partition=='student_test').sum()),
      'total_gps_observations':int(m.observations.sum()),
      'total_road_nodes_across_trip_networks':int(m.nodes.sum()),
      'total_road_arcs_across_trip_networks':int(m.arcs.sum()),
      'invalid_gps_observations':int(m.invalid_observations.sum()),
      'missing_or_invalid_timestamps':int(m.missing_or_invalid_timestamp.sum()),
      'duplicate_timestamps':int(m.duplicate_timestamp.sum()),
      'nonmonotonic_timestamps':int(m.nonmonotonic_timestamp.sum()),
      'large_gps_time_gaps':int(m.temporal_gaps.sum()),
      'abnormal_speed_jumps_flagged':int(m.abnormal_jumps.sum()),
      'invalid_road_arcs':int(m.invalid_road_arcs.sum()),
      'zero_length_road_arcs':int(sum(q['zero_length_arcs'] for q in roadq)),
      'invalid_road_endpoint_references':int(sum(q['invalid_endpoint_arcs'] for q in roadq)),
      'candidate_rows':int(m.candidate_rows.sum()),
      'observations_with_candidates':int(m.candidate_observations.sum()),
      'observations_without_candidates':int(m.missing_candidate_observations.sum()),
      'fallback_candidate_rows':int(m.fallback_candidate_rows.sum()),
      'coverage_fraction':float(m.candidate_observations.sum()/m.observations.sum()),
      'weak_proxy_training_observations':int(nweak['train']),
      'weak_proxy_training_coverage_fraction':float(nweak['train']/m.loc[m.partition=='train','observations'].sum()),
      'verified_observation_ground_truth_count':0,
      'invalid_reference_arc_ids_in_train':int(sum(q.get('invalid_route_arc_ids',0) for q in labq)),
      'broken_directed_train_reference_connections':int(sum(q.get('broken_route_connections',0) for q in labq)),
      'training_weak_candidate_misses':int(sum(q.get('weak_candidate_misses',0) for q in labq)),
      'test_reference_derived_labels_used':False,
      'limitations':['Organizer PDFs not supplied','Per-point ground truth unavailable; weak alignment is NOT verified',
      'CSV.gz instead of Parquet because pyarrow unavailable in current runtime',
      'Candidate coverage includes explicit bounded nearest-road fallback']}
    # Validate observed raw-files presence without rereading entire raw datasets.
    raw_dirs=discover(data_root)
    missing=[]
    for part,tid,path in raw_dirs:
        for ext in ('track','nodes','arcs','route'):
            if not (path/f'{tid}.{ext}').is_file():missing.append(f'{part}/{tid}/{tid}.{ext}')
    geodata['missing_raw_files']=missing
    geodata['raw_trip_folder_count']=len(raw_dirs)
    if len(raw_dirs)!=len(m):errors.append('Raw trip directories and processed manifest count mismatch')
    if missing:errors.append('Some raw files are missing')
    if len(m)!=90:errors.append('Expected 90 trip folders based on supplied archive inventory')
    geodata['audit_status']='passed' if not errors else 'failed'
    geodata['audit_failures']=errors
    write_json(geodata,out/'data_quality_report.json')
    write_json({'status':geodata['audit_status'],'errors':errors,'actual_aggregate_rows':nkeys,
                'actual_weak_training_observations':nweak['train'],
                'training_inference_parity_on_real_trip':not any('training/inference' in e for e in errors)},out/'aggregate_audit_report.json')
    return geodata
