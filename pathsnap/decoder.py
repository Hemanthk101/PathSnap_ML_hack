"""Directed road-network transitions and log-space Viterbi decoding."""
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 36.0
    emission_weight: float = 0.85
    restart_penalty: float = 8.0
    max_gap_s: float = 45.0
    reverse_jitter_m: float = 12.0


@dataclass
class Match:
    observation_arcs: np.ndarray
    candidate_rows: np.ndarray
    arc_chains: list
    route_lines: list
    summary: dict


class DirectedGraph:
    def __init__(self, road):
        self.endpoints = road['arc_raw']
        self.lengths = road['lengths']
        self.node_xy = road['coords_xy']
        self.components = road['nodes'].weak_component.to_numpy()
        self.edges = defaultdict(list)
        for arc in np.flatnonzero(road['usable']):
            start, end = self.endpoints[arc]
            self.edges[int(start)].append((int(end), float(self.lengths[arc]), int(arc)))
        self.cache = {}

    def search(self, source, limit):
        key = (source, limit)
        if key in self.cache:
            return self.cache[key]
        distances = {source: 0.0}
        previous = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for endpoint, length, arc in self.edges.get(node, ()):
                total = distance + length
                if total <= limit and total < distances.get(endpoint, np.inf):
                    distances[endpoint] = total
                    previous[endpoint] = (node, arc)
                    heappush(queue, (total, endpoint))
        if len(self.cache) >= 384:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (distances, previous)
        return distances, previous

    def connection(self, arc_a, frac_a, arc_b, frac_b, limit, jitter, return_path=False):
        length_a, length_b = self.lengths[arc_a], self.lengths[arc_b]
        if arc_a == arc_b:
            along = (frac_b - frac_a) * length_a
            if -jitter <= along <= limit:
                return max(0.0, along), ()
        tail, head = int(self.endpoints[arc_a, 1]), int(self.endpoints[arc_b, 0])
        ends = (1.0 - frac_a) * length_a + frac_b * length_b
        if ends > limit or self.components[tail] != self.components[head]:
            return np.inf, ()
        if tail == head:
            return float(ends), None if arc_a == arc_b else ()
        if np.linalg.norm(self.node_xy[tail] - self.node_xy[head]) > limit - ends:
            return np.inf, ()
        bucket = min(size for size in (120, 240, 480, 960, 1920, 3200) if size >= limit)
        distances, previous = self.search(tail, bucket)
        distance = ends + distances.get(head, np.inf)
        if distance > limit:
            return np.inf, ()
        path = []
        if return_path:
            node = head
            while node != tail:
                node, arc = previous[node]
                path.append(arc)
        return float(distance), (tuple(reversed(path)) or None) if arc_a == arc_b and return_path else tuple(reversed(path))


class Decoder:
    def __init__(self, prepared, config=DecoderConfig()):
        self.observations = prepared['observations']
        self.road = prepared['road']
        self.count = len(self.observations)
        self.config = config
        self.candidates = prepared['candidates']
        self.arcs = self.candidates.arc_id.to_numpy(dtype=int)
        self.fractions = self.candidates.projection_fraction.to_numpy(dtype=float)
        self.snap = self.candidates[['snapped_longitude', 'snapped_latitude']].to_numpy(dtype=float)
        self.nodes = self.road['nodes'][['longitude', 'latitude']].to_numpy(dtype=float)
        self.graph = DirectedGraph(self.road)
        self.groups = [np.empty(0, dtype=int) for _ in range(self.count)]
        for obs, rows in self.candidates.groupby('observation_index', sort=False).indices.items():
            self.groups[obs] = rows[np.argsort(self.arcs[rows], kind='stable')]
        self.dt = self.observations.delta_t_s.to_numpy(dtype=float)
        self.step = self.observations.distance_from_previous_m.to_numpy(dtype=float)
        self.distances = self.transition_distances()

    def limit(self, observation):
        return min(2400.0, max(110.0, self.step[observation] * 1.9 + 55.0,
                               self.dt[observation] * 31.0 + 35.0))

    def transition_distances(self):
        distances = [None] * self.count
        for obs in range(1, self.count):
            left, right = self.groups[obs - 1], self.groups[obs]
            if not len(left) or not len(right) or not 0 < self.dt[obs] <= self.config.max_gap_s:
                continue
            if not np.isfinite(self.step[obs]):
                continue
            limit = self.limit(obs)
            matrix = np.full((len(left), len(right)), np.inf)
            for i, old_row in enumerate(left):
                for j, new_row in enumerate(right):
                    matrix[i, j], _ = self.connection(old_row, new_row, limit)
            distances[obs] = matrix
        return distances

    def connection(self, old_row, new_row, limit, return_path=False):
        return self.graph.connection(self.arcs[old_row], self.fractions[old_row],
                                     self.arcs[new_row], self.fractions[new_row],
                                     limit, self.config.reverse_jitter_m, return_path)

    def decode(self, probabilities, config=None, independent=False):
        cfg = config or self.config
        if (cfg.max_gap_s, cfg.reverse_jitter_m) != (self.config.max_gap_s, self.config.reverse_jitter_m):
            raise ValueError('Rebuild transitions when gap or jitter configuration changes')
        p = np.asarray(probabilities, dtype=float)
        if p.shape != self.arcs.shape or not np.isfinite(p).all():
            raise ValueError('One finite probability is required per candidate row')
        selected = np.full(self.count, -1, dtype=int)
        linked = np.zeros(self.count, dtype=bool)
        if independent:
            for obs, rows in enumerate(self.groups):
                if len(rows):
                    selected[obs] = rows[p[rows].argmax()]
            for obs, matrix in enumerate(self.distances):
                if matrix is not None:
                    a, b = np.searchsorted(self.arcs[self.groups[obs - 1]], self.arcs[selected[obs - 1]]), np.searchsorted(
                        self.arcs[self.groups[obs]], self.arcs[selected[obs]])
                    linked[obs] = np.isfinite(matrix[a, b])
        else:
            back, connecting, best = [None] * self.count, [None] * self.count, [-1] * self.count
            score = None
            for obs, rows in enumerate(self.groups):
                if not len(rows):
                    score = None
                    continue
                local = np.clip(p[rows], 1e-8, 1.0)
                emission = np.log(local / local.sum()) * cfg.emission_weight
                matrix = self.distances[obs]
                if matrix is None:
                    score = emission
                else:
                    beta = cfg.beta_m * (1.0 + min(self.dt[obs], 25.0) / 18.0)
                    # The transition potential penalizes disagreement with observed geodesic displacement.
                    transition = -np.abs(matrix - self.step[obs]) / beta
                    values = score[:, None] + transition
                    parent = values.argmax(axis=0)
                    highest = values[parent, np.arange(len(rows))]
                    restart_from = int(score.argmax())
                    restart_score = score[restart_from] - cfg.restart_penalty
                    use_link = highest >= restart_score
                    back[obs] = np.where(use_link, parent, restart_from)
                    connecting[obs] = use_link
                    score = emission + np.maximum(highest, restart_score)
                score -= score.max()
                best[obs] = int(score.argmax())
            current = -1
            for obs in range(self.count - 1, -1, -1):
                if best[obs] < 0:
                    current = -1
                    continue
                if current < 0:
                    current = best[obs]
                selected[obs] = self.groups[obs][current]
                if back[obs] is None:
                    current = -1
                else:
                    linked[obs] = connecting[obs][current]
                    current = back[obs][current]
        return self.build_match(selected, linked)

    def build_match(self, selected, linked):
        observation_arcs = np.full(self.count, -1, dtype=int)
        chains, lines = [], []
        matched = np.flatnonzero(selected >= 0)
        route_length = 0.0
        for obs in matched:
            row = selected[obs]
            arc = int(self.arcs[row])
            observation_arcs[obs] = arc
            coord = self.snap[row].tolist()
            if not linked[obs]:
                chains.append([arc])
                lines.append([coord])
                continue
            old_row = selected[obs - 1]
            old_arc = int(self.arcs[old_row])
            distance, bridge = self.connection(old_row, row, self.limit(obs), return_path=True)
            route_length += distance
            if arc != old_arc or bridge is None or bridge:
                bridge = bridge or ()
                chains[-1].extend((*bridge, arc))
                lines[-1].append(self.nodes[self.graph.endpoints[old_arc, 1]].tolist())
                lines[-1].extend(self.nodes[self.graph.endpoints[part, 1]].tolist() for part in bridge)
            # Suppress tiny backward projection jitter in rendered route geometry.
            if distance > 1e-6 or arc != old_arc:
                lines[-1].append(coord)
        eligible_links = sum(matrix is not None for matrix in self.distances)
        link_count = int(linked.sum())
        return Match(observation_arcs, selected, chains, lines, {
            'observations': self.count, 'matched': len(matched), 'route_km': route_length / 1000.0,
            'route_fragments': len(chains), 'connected_pairs': link_count,
            'eligible_pairs': eligible_links, 'continuity': link_count / eligible_links if eligible_links else None,
            'disconnected_pairs': eligible_links - link_count,
            'time_breaks': int(np.sum(np.isfinite(self.dt) & ((self.dt <= 0) | (self.dt > self.config.max_gap_s)))),
        })
