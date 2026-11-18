"""Exercise each available selected partition using live HTTP inference."""
import argparse
import json
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(base, endpoint, body=None):
    args = {} if body is None else {'data': json.dumps(body).encode(), 'headers': {'Content-Type': 'application/json'}}
    with urlopen(Request(base.rstrip('/') + endpoint, **args), timeout=120) as response:
        return response.status, json.load(response)


def verify(base, partition, expect, expected_metrics=None, require_reference=False):
    for attempt in range(15):
        try:
            _, health = request(base, '/api/health')
            break
        except URLError:
            if attempt == 14:
                raise
            sleep(1)
    assert health['status'] == 'ready'
    _, catalog = request(base, '/api/trips')
    trips = next(group['trips'] for group in catalog['partitions'] if group['key'] == partition)
    assert len(trips) == expect, f'{len(trips)} trips available, expected {expect}'
    rows = []
    for trip in trips:
        status, data = request(base, '/api/match', {'partition': partition, 'trip_id': trip['id']})
        metrics = data['metrics']['hmm']
        assert status == 200 and data['trip_id'] == trip['id'] and data['partition'] == partition
        assert len(data['map']['observations']) == metrics['observations'] == trip['observations']
        assert sum(row['arc'] is not None for row in data['map']['observations']) == metrics['matched']
        assert len(data['map']['route']) == metrics['route_fragments']
        assert set(data['map']) == {'bounds', 'network', 'route', 'classifier', 'observations'}
        assert all(not {'reference', 'reference_arc', 'aligned_arc_id'} & obs.keys() for obs in data['map']['observations'])
        if partition == 'student_test' or require_reference:
            assert data['reference_metrics_available']
        if expected_metrics is not None:
            expected = expected_metrics[trip['id']]
            for key in ('matched', 'proxy_correct', 'proxy_observations', 'connected_pairs', 'eligible_pairs',
                        'route_fragments', 'ordered_overlap_arcs', 'sequence_f1'):
                assert metrics[key] == expected[key], f'{trip["id"]} metric differs from route experiment: {key}'
        continuity = f'{metrics["continuity"]:.1%}' if metrics['continuity'] is not None else 'N/A'
        print(f'HTTP {status} POST /api/match  {partition}/{trip["id"]}  '
              f'GPS={metrics["observations"]}  continuity={continuity}  '
              f'time={data["timing"]["total_ms"]} ms', flush=True)
        rows.append({'trip_id': trip['id'], 'status': status, 'timing': data['timing'], 'metrics': metrics,
                     'classifier_metrics': data['metrics']['classifier']})
    try:
        request(base, '/api/match', {'partition': partition, 'trip_id': '99999999'})
    except HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError('Unknown trip must return 404')
    print('Verified all requests and unknown-trip handling.', flush=True)
    return {'partition': partition, 'model': health['model'], 'trips': rows, 'unknown_trip_status': 404,
            'checked_against_route_experiment': expected_metrics is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--partition', default='student_test')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compare', type=Path, help='Route comparison JSON for exact per-trip metric checks')
    parser.add_argument('--require-reference', action='store_true', help='Require private reference metrics for every trip')
    args = parser.parse_args()
    expected_metrics = None
    if args.compare is not None:
        expected_metrics = {row['trip_id']: row for row in json.loads(
            args.compare.read_text(encoding='utf-8'))['per_trip']['hmm']}
    result = verify(args.url, args.partition, args.count, expected_metrics, args.require_reference)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
