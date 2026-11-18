from dataclasses import replace

import numpy as np
import pandas as pd

from pathsnap.decoder import Decoder, DecoderConfig
from pathsnap.evaluation import ordered_overlap


def setup_trip():
    # Directed chain 0 → 1 → 2 → 3, plus disconnected arc 4 → 5.
    arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 5]])
    xy = np.array([[0, 0], [20, 0], [40, 0], [60, 0], [0, 25], [20, 25]])
    nodes = pd.DataFrame({'longitude': xy[:, 0] / 100000, 'latitude': xy[:, 1] / 100000,
                          'weak_component': [0, 0, 0, 0, 1, 1]})
    road = {'arc_raw': arcs, 'lengths': np.full(4, 20.), 'coords_xy': xy, 'nodes': nodes,
            'usable': np.ones(4, dtype=bool)}
    points = pd.DataFrame({'observation_index': [0, 1, 1, 3, 4], 'arc_id': [0, 2, 3, 0, 2],
                           'projection_fraction': [.5] * 5,
                           'snapped_longitude': [0.0001, 0.0005, 0.0001, 0.0001, 0.0005],
                           'snapped_latitude': [0, 0, 0.00025, 0, 0]})
    observations = pd.DataFrame({'delta_t_s': [np.nan, 2, 1, 2, 120],
                                 'distance_from_previous_m': [np.nan, 40, 0, 40, 40]})
    return {'candidates': points, 'road': road, 'observations': observations}


def test_directed_bridge_overrides_independent_outlier_and_preserves_missing_rows():
    decoder = Decoder(setup_trip())
    p = np.array([.99, .3, .9, .8, .8])
    raw = decoder.decode(p, independent=True)
    result = decoder.decode(p)
    assert raw.observation_arcs.tolist() == [0, 3, -1, 0, 2]
    assert result.observation_arcs.tolist() == [0, 2, -1, 0, 2]
    assert result.arc_chains == [[0, 1, 2], [0], [2]]
    assert result.summary['connected_pairs'] == result.summary['eligible_pairs'] == 1
    assert result.summary['time_breaks'] == 1
    assert len(result.route_lines[0]) == 4


def test_disconnected_candidates_restart_instead_of_creating_a_line():
    trip = setup_trip()
    trip['candidates'] = trip['candidates'].drop(1).reset_index(drop=True)
    result = Decoder(trip).decode([.9, .9, .9, .9])
    assert result.summary['disconnected_pairs'] == 1
    assert all(len(line) == 1 for line in result.route_lines)


def test_transition_respects_direction_and_allows_small_projection_jitter():
    graph = Decoder(setup_trip()).graph
    assert graph.connection(2, .5, 0, .5, 110, 12)[0] == np.inf
    assert graph.connection(1, .5, 1, .4, 110, 12)[0] == 0
    assert graph.connection(1, 1.0, 1, 0.0, 110, 12)[0] == np.inf


def test_empty_candidate_groups_return_unmatched_observations():
    trip = setup_trip()
    trip['candidates'] = trip['candidates'].iloc[0:0]
    result = Decoder(trip).decode([])
    assert result.observation_arcs.tolist() == [-1] * 5
    assert result.route_lines == []
    assert result.summary['continuity'] is None


def test_decoder_rejects_config_changes_that_require_new_transitions():
    decoder = Decoder(setup_trip())
    with np.testing.assert_raises(ValueError):
        decoder.decode(np.ones(5), replace(DecoderConfig(), max_gap_s=5))


def test_ordered_overlap_handles_loops_and_reversed_order():
    assert ordered_overlap([1, 2, 1, 4], [4, 1, 2, 1]) == 3
    assert ordered_overlap([], [1]) == 0
