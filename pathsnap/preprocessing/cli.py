"""CLI: python -m pathsnap.preprocessing.cli --help"""
import argparse,json,logging
from pathlib import Path
import pandas as pd
from .pipeline import prepare_training_dataset,process_one_trip,feature_schema
from .io import discover,write_json
from .config import Config

def main():
    p=argparse.ArgumentParser(description='PathSnap auditable preprocessing')
    p.add_argument('command',choices=['inventory','process','validate','benchmark','infer'])
    p.add_argument('--data-root',default='data/raw',help='Folder containing train/ and student_test/')
    p.add_argument('--out',default='data/processed')
    p.add_argument('--partition',choices=['train','student_test'])
    p.add_argument('--trip',help='Trip folder for infer/quick benchmark')
    p.add_argument('--config',help='JSON preprocessing config')
    p.add_argument('--limit',type=int,help='Process first N trips (smoke tests only)')
    p.add_argument('--allow-student-references',action='store_true',help='Only use when official rules explicitly permit student-test reference access')
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    cfg=Config.from_path(a.config)
    if a.command=='inventory':
        d=discover(a.data_root,a.partition)
        info=[{'partition':part,'trip_id':id,'files_present':sorted([x.suffix for x in path.iterdir() if x.is_file()])} for part,id,path in d]
        write_json({'count':len(d),'trip_inventory':info},Path(a.out)/'raw_files_inventory.json')
        print(f'Discovered {len(d)} trip directories; wrote raw_files_inventory.json')
    elif a.command in ('process','benchmark'):
        if a.command=='benchmark' and a.trip:
            r=process_one_trip(a.trip,'benchmark',None,cfg,False)
            print(json.dumps(r['manifest'],indent=2));return
        r=prepare_training_dataset(a.data_root,a.out,cfg,a.partition,a.allow_student_references,a.limit)
        print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
        if r['errors']:raise SystemExit(1)
    elif a.command=='validate':
        from .audit import audit_processed
        report=audit_processed(a.data_root,a.out)
        print(json.dumps({'status':report['audit_status'],'failures':report['audit_failures'],
                          'observations':report['total_gps_observations']},indent=2))
        if report['audit_status']!='passed':raise SystemExit(1)
    else:
        if not a.trip: p.error('--trip is mandatory for infer')
        r=process_one_trip(a.trip,'inference',Path(a.out),cfg,use_reference=False)
        print(json.dumps(r['manifest'],indent=2))

if __name__=='__main__':main()
