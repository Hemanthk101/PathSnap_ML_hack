"""Strict file parsing, schema discovery, and CSV/Parquet storage adapters."""
from pathlib import Path
import importlib.util, json, gzip
import numpy as np
import pandas as pd

HAS_ARROW = importlib.util.find_spec('pyarrow') is not None

class DataError(ValueError):
    pass

def read_numeric(path: Path, columns: int, integer=False):
    path=Path(path)
    if not path.is_file():
        raise DataError(f'Missing required file: {path}')
    try:
        a = np.loadtxt(path, dtype=np.int64 if integer else np.float64, ndmin=2)
    except (ValueError, OSError) as e:
        raise DataError(f'Failed reading {path}: {e}') from e
    if a.shape[1] != columns:
        raise DataError(f'{path}: expected {columns} columns, got {a.shape[1]}')
    if len(a) == 0:
        raise DataError(f'{path}: empty file')
    return a

def load_trip(trip_path, *, require_route=False, load_route=True):
    p=Path(trip_path)
    tid=p.name
    got = {}
    for ext, n, integer in (('track',3,False),('nodes',2,False),('arcs',2,True)):
        got[ext] = read_numeric(p / f'{tid}.{ext}', n, integer)
    route_path = p / f'{tid}.route'
    if load_route and route_path.is_file():
        got['route'] = read_numeric(route_path,1,True).ravel()
    elif require_route:
        raise DataError(f'Missing reference route: {route_path}')
    else:
        got['route'] = None
    got['trip_id']=tid
    return got

def discover(data_root, partition=None):
    root=Path(data_root)
    # Accept either an extracted folder containing train/ and student_test/ or a parent.
    if not (root/'train').exists():
        matches=[p for p in root.glob('*/train') if p.is_dir()]
        if len(matches)==1:
            root=matches[0].parent
    parts=[partition] if partition else ['train','student_test']
    entries=[]
    for part in parts:
        p=root/part
        if not p.is_dir():
            raise DataError(f'Partition not found: {p}')
        for d in sorted(p.iterdir()):
            if d.is_dir():
                entries.append((part,d.name,d))
    return entries

def write_table(df, base, *, index=False):
    """Parquet if available; otherwise truthful .csv.gz fallback (never fake Parquet)."""
    base=Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    if HAS_ARROW:
        p=base.with_suffix('.parquet')
        df.to_parquet(p,index=index,compression='zstd')
    else:
        p=base.with_suffix('.csv.gz')
        df.to_csv(p,index=index,compression={'method':'gzip','compresslevel':1},float_format='%.12g')
    return p

def read_table(base):
    base=Path(base)
    for p in [base.with_suffix('.parquet'),base.with_suffix('.csv.gz')]:
        if p.is_file():
            return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,dtype={'trip_id':str})
    raise FileNotFoundError(f'No processed table at {base} (.parquet or .csv.gz)')

def write_json(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False,default=_json_default)
    return path

def _json_default(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o) if np.isfinite(o) else None
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,Path): return str(o)
    raise TypeError(f'Not JSON serializable: {type(o).__name__}')
