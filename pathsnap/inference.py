"""Inference and map serialization with no access to reference routes."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .decoder import Decoder, DecoderConfig
from .preprocessing import prepare_inference_trip
from .preprocessing.config import Config
from .preprocessing.geometry import FEATURES


class MapMatcher:
    def __init__(self, model_dir):
        directory = Path(model_dir)
        self.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['features'] != FEATURES:
            raise ValueError('Model feature schema does not match the inference pipeline')
        config = self.manifest['preprocessing'].copy()
        config.pop('schema_version')
        self.config = Config(**config).validate()
        self.decoder_config = DecoderConfig(**json.loads((directory / 'decoder.json').read_text(encoding='utf-8')))
        self.model = joblib.load(directory / self.manifest['artifact'])

    def predict(self, trip_path):
        start = perf_counter()
        prepared = prepare_inference_trip(trip_path, self.config)
        clean_s = perf_counter() - start
        features, candidates = prepared['features'], prepared['candidates']
        keys = ['observation_index', 'arc_id']
        if not features[keys].equals(candidates[keys]):
            raise ValueError('Candidate and feature order differs')
        start = perf_counter()
        probabilities = self.model.predict_proba(features[FEATURES])[:, 1] if len(features) else np.empty(0)
        score_s = perf_counter() - start
        start = perf_counter()
        decoder = Decoder(prepared, self.decoder_config)
        hmm = decoder.decode(probabilities)
        baseline = decoder.decode(probabilities, independent=True)
        decoder_s = perf_counter() - start
        return prepared, probabilities, hmm, baseline, {
            'preprocessing_ms': round(1000 * clean_s), 'classifier_ms': round(1000 * score_s),
            'decoder_ms': round(1000 * decoder_s),
        }


def number(value, precision=2):
    return round(float(value), precision) if np.isfinite(value) else None


def map_payload(prepared, probabilities, hmm, baseline):
    obs, candidates, road = prepared['observations'], prepared['candidates'], prepared['road']
    snaps = candidates[['snapped_longitude', 'snapped_latitude']].to_numpy()
    offsets = candidates.distance_m.to_numpy()
    gps, observations = [], []
    for index, row in enumerate(obs.itertuples(index=False)):
        raw = [number(row.longitude, 7), number(row.latitude, 7)] if row.valid_gps else None
        gps.append(raw)
        match = hmm.candidate_rows[index]
        observations.append({
            'raw': raw, 'time': number(row.timestamp_s, 3), 'speed_mps': number(row.speed_mps),
            'arc': int(hmm.observation_arcs[index]) if match >= 0 else None,
            'classifier_arc': int(baseline.observation_arcs[index]) if baseline.candidate_rows[index] >= 0 else None,
            'snapped': np.round(snaps[match], 7).tolist() if match >= 0 else None,
            'offset_m': number(offsets[match], 1) if match >= 0 else None,
            'candidate_score': number(probabilities[match], 4) if match >= 0 else None,
        })
    coords = np.asarray([point for point in gps if point is not None], dtype=float)
    nodes = road['nodes'][['longitude', 'latitude']].to_numpy()
    edges = road['arc_raw'][road['usable']]
    if len(coords):
        low, high = coords.min(axis=0), coords.max(axis=0)
    else:
        bounds_source = nodes[np.isfinite(nodes).all(axis=1)]
        low, high = bounds_source.min(axis=0), bounds_source.max(axis=0)
    padding = np.maximum(0.003, (high - low) * 0.30)
    nearby = ((nodes >= low - padding) & (nodes <= high + padding)).all(axis=1)
    edges = edges[nearby[edges].any(axis=1)]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    if len(edges) > 22000:
        edges = edges[np.linspace(0, len(edges) - 1, 22000, dtype=int)]
    vertex_ids, edge_indices = np.unique(edges.ravel(), return_inverse=True)
    return {
        'bounds': [low.tolist(), high.tolist()],
        'network': {'vertices': np.round(nodes[vertex_ids], 7).tolist(), 'edges': edge_indices.reshape(-1, 2).tolist()},
        'route': [np.round(line, 7).tolist() for line in hmm.route_lines],
        'classifier': [np.round(line, 7).tolist() for line in baseline.route_lines],
        'observations': observations,
    }
