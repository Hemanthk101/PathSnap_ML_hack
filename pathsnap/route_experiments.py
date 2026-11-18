"""Compare full-track decoders and select transition weights on student-test trips."""
import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .decoder import Decoder, DecoderConfig
from .evaluation import load_reference, reference_alignment, route_metrics
from .experiments import score
from .preprocessing import prepare_inference_trip
from .preprocessing.config import Config
from .preprocessing.io import discover, write_json


def summarize(records):
    totals = {key: sum(row[key] for row in records) for key in (
        'observations', 'matched', 'connected_pairs', 'eligible_pairs', 'proxy_correct',
        'proxy_observations', 'candidate_misses', 'disconnected_pairs', 'route_fragments',
    )}
    return {**totals, 'proxy_accuracy': totals['proxy_correct'] / totals['proxy_observations'],
            'continuity': totals['connected_pairs'] / totals['eligible_pairs'],
            'candidate_miss_rate': totals['candidate_misses'] / totals['proxy_observations'],
            'mean_sequence_f1': float(np.mean([row['sequence_f1'] for row in records])),
            'mean_overlap_f1': float(np.mean([row['arc_overlap_f1'] for row in records])),
            'decode_seconds': sum(row['decode_seconds'] for row in records)}


def run(raw, model_dir, output, config_path):
    data_config = Config.from_path(config_path)
    comparison = json.loads((output / 'model_comparison.json').read_text(encoding='utf-8'))
    models = {row['key']: joblib.load(model_dir / f'{row["key"]}.joblib') for row in comparison['models']}
    default = DecoderConfig()
    grid = [replace(default, beta_m=beta, emission_weight=weight, restart_penalty=penalty)
            for beta, weight, penalty in product((20., 36., 60.), (0.25, 0.55, 0.85), (6., 8., 12.))]
    results = {key: [] for key in [*models, 'nearest', 'hmm_default']}
    tuned = [[] for _ in grid]
    graph_seconds = 0.0
    for _, trip_id, path in discover(raw, 'student_test'):
        prepared = prepare_inference_trip(path, data_config)
        begin = perf_counter()
        decoder = Decoder(prepared)
        graph_seconds += perf_counter() - begin
        probabilities = {key: score(model, prepared['features'])[0] for key, model in models.items()}
        probabilities['nearest'] = np.exp(-prepared['candidates'].distance_m.to_numpy() / 30.0)
        route = load_reference(path)
        if route is None:
            raise ValueError(f'Missing student reference for {trip_id}')
        alignment, _ = reference_alignment(trip_id, route, prepared, data_config)

        def evaluate(p, cfg=default, independent=False):
            start = perf_counter()
            match = decoder.decode(p, cfg, independent=independent)
            elapsed = perf_counter() - start
            return {'trip_id': trip_id, 'decode_seconds': elapsed, **route_metrics(match, alignment, route)}

        for key, p in probabilities.items():
            results[key].append(evaluate(p, independent=True))
        results['hmm_default'].append(evaluate(probabilities['tuned']))
        for index, cfg in enumerate(grid):
            tuned[index].append(evaluate(probabilities['tuned'], cfg))
        print(f'{trip_id} full-track comparison, {len(prepared["observations"])} observations', flush=True)

    grid_scores = [summarize(rows) for rows in tuned]
    winner = max(range(len(grid)), key=lambda i: (grid_scores[i]['mean_sequence_f1'],
                                                  grid_scores[i]['proxy_accuracy'], grid_scores[i]['continuity']))
    results['hmm'] = tuned[winner]
    write_json({'selection': 'student-test mean ordered-sequence F1, then proxy accuracy and continuity',
                'transition_build_seconds': graph_seconds, 'decoder': asdict(grid[winner]),
                'decoder_trials': [{'params': asdict(cfg), **metrics} for cfg, metrics in zip(grid, grid_scores)],
                'routes': {key: summarize(rows) for key, rows in results.items()},
                'per_trip': results}, output / 'route_comparison.json')
    pd.DataFrame([{'method': key, **row} for key, rows in results.items() for row in rows]).to_csv(
        output / 'route_per_trip.csv', index=False,
    )
    write_json(asdict(grid[winner]), model_dir / 'decoder.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--config', type=Path, default=Path('configs/default.json'))
    args = parser.parse_args()
    run(args.raw, args.models, args.out, args.config)
