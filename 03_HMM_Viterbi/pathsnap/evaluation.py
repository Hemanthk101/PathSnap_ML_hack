"""Reference-only evaluation. Inference does not import this module."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from .preprocessing.io import read_numeric
from .preprocessing.route_alignment import analyze_reference, derive_weak_targets


def load_reference(trip_path, reference_root=None):
    trip_path = Path(trip_path)
    trip_id = trip_path.name
    directory = Path(reference_root) if reference_root is not None else trip_path
    flat = directory / f'{trip_id}.route'
    nested = directory / trip_id / f'{trip_id}.route'
    path = flat if flat.is_file() else nested
    return read_numeric(path, 1, integer=True).ravel() if path.is_file() else None


def reference_alignment(trip_id, route, prepared, config):
    alignment, _, quality = analyze_reference(
        {'trip_id': trip_id, 'route': route}, prepared['road'], prepared['candidates'],
        prepared['observations'], config,
    )
    return alignment, quality


def candidate_metrics(rows, probabilities):
    y = rows['weak_target'].to_numpy()
    predicted = probabilities >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average='macro', labels=[0, 1], zero_division=0,
    )
    leaders = rows.assign(score=probabilities).sort_values(
        ['score', 'arc_id'], ascending=[False, True], kind='stable',
    ).drop_duplicates(['trip_id', 'observation_index'])
    return {
        'accuracy': accuracy_score(y, predicted), 'macro_precision': precision,
        'macro_recall': recall, 'macro_f1': f1, 'top1': float(leaders.weak_target.mean()),
        'log_loss': log_loss(y, probabilities, labels=[0, 1]),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'candidate_rows': len(rows), 'aligned_observations': len(leaders),
    }


def labeled_candidates(prepared, alignment):
    targets = derive_weak_targets(prepared['candidates'], alignment)
    keys = ['partition', 'trip_id', 'observation_index', 'arc_id']
    return prepared['features'].merge(targets[keys + ['weak_target']], on=keys, validate='one_to_one')


def collapse_arcs(arcs):
    result = []
    for arc in arcs:
        if not result or result[-1] != arc:
            result.append(int(arc))
    return result


def ordered_overlap(predicted, reference):
    """Length of the longest common subsequence via bit-parallel dynamic programming."""
    masks = {}
    for index, arc in enumerate(reference):
        masks[arc] = masks.get(arc, 0) | (1 << index)
    state = 0
    for arc in predicted:
        union = state | masks.get(arc, 0)
        state = union & ~(union - ((state << 1) | 1))
    return state.bit_count()


def route_metrics(match, alignment, route):
    summary = dict(match.summary)
    if route is None:
        return summary
    eligible = alignment[alignment.alignment_status.isin(['weak_supported', 'weak_candidate_miss'])]
    covered = eligible.alignment_status.eq('weak_supported')
    indices = eligible.observation_index.to_numpy(dtype=int)
    correct = int((match.observation_arcs[indices] == eligible.aligned_arc_id.to_numpy()).sum())
    sequence = [arc for chain in match.arc_chains for arc in collapse_arcs(chain)]
    reference = collapse_arcs(route)
    overlap = ordered_overlap(sequence, reference)
    shared = sum((Counter(sequence) & Counter(reference)).values())
    summary.update({
        'proxy_correct': correct, 'proxy_observations': len(eligible),
        'proxy_accuracy': correct / len(eligible) if len(eligible) else None,
        'candidate_misses': int((~covered).sum()),
        'candidate_miss_rate': float((~covered).mean()) if len(eligible) else None,
        'sequence_f1': 2 * overlap / (len(sequence) + len(reference)) if sequence or reference else None,
        'ordered_overlap_arcs': overlap, 'reference_arcs': len(reference), 'predicted_arcs': len(sequence),
        'arc_overlap_f1': 2 * shared / (len(sequence) + len(reference)) if sequence or reference else None,
    })
    return summary
