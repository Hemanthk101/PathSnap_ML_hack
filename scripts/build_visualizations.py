"""Generate EDA, measured model plots and a Folium map from the student data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
from branca.element import Element
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation import load_reference
from pathsnap.experiments import estimator, load_training
from pathsnap.inference import MapMatcher, map_payload
from pathsnap.preprocessing.geometry import FEATURES, GEOD
from pathsnap.preprocessing.io import discover, read_numeric, write_json

INK, GREEN, TEAL, SAND = '#183b2c', '#346953', '#7aac83', '#b95e38'
MODEL_LABELS = ['Logistic', 'Gaussian NB', 'Decision tree', 'Random forest', 'XGBoost', 'XGBoost tuned']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': '#bbccc1'})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name, plots):
    fig.savefig(plots / f'{name}.png', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def explore(raw, processed, output, plots):
    rows, interval_groups = [], {'train': [], 'student_test': []}
    for partition, trip_id, folder in discover(raw):
        track = read_numeric(folder / f'{trip_id}.track', 3)
        lon, lat, times = track.T
        valid = np.isfinite(track).all(axis=1) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
        with_data = track[valid]
        dt = np.diff(times)
        interval_groups[partition].append(dt[np.isfinite(dt)])
        _, _, steps = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
        rows.append({'partition': partition, 'trip_id': trip_id, 'gps_rows': len(track),
                     'invalid_gps': int((~valid).sum()), 'track_km': np.nansum(steps) / 1000,
                     'median_step_s': float(np.nanmedian(dt)) if len(dt) else None,
                     'max_gap_s': float(np.nanmax(dt)) if len(dt) else None,
                     'west': with_data[:, 0].min() if len(with_data) else None,
                     'south': with_data[:, 1].min() if len(with_data) else None,
                     'east': with_data[:, 0].max() if len(with_data) else None,
                     'north': with_data[:, 1].max() if len(with_data) else None,
                     **{f'{ext}_bytes': (folder / f'{trip_id}.{ext}').stat().st_size
                        if (folder / f'{trip_id}.{ext}').is_file() else 0
                        for ext in ['track', 'nodes', 'arcs', 'route']}})
    raw_stats = pd.DataFrame(rows)
    network = pd.DataFrame(read_json(processed / 'road_network_quality.json')['per_trip'])
    manifest = pd.read_csv(processed / 'dataset_manifest.csv', dtype={'trip_id': str})
    audit = raw_stats.merge(network, on=['partition', 'trip_id'], validate='one_to_one').merge(
        manifest[['partition', 'trip_id', 'candidate_rows', 'fallback_candidate_rows', 'temporal_gaps',
                  'duplicate_timestamp', 'nonmonotonic_timestamp', 'abnormal_jumps']],
        on=['partition', 'trip_id'], validate='one_to_one',
    )
    audit.to_csv(output / 'dataset_by_trip.csv', index=False)
    validation = read_json(processed / 'dataset_validation_report.json')['per_trip']
    combined = {}
    for partition, group in audit.groupby('partition'):
        dt = np.concatenate(interval_groups[partition])
        sums = group[['gps_rows', 'invalid_gps', 'isolated_nodes', 'invalid_arcs', 'node_count', 'arc_count',
                      'candidate_rows', 'fallback_candidate_rows', 'temporal_gaps', 'abnormal_jumps',
                      'duplicate_timestamp', 'nonmonotonic_timestamp']].sum().to_dict()
        missing = {name: sum(row['feature_nan_counts'][name] for row in validation if row['partition'] == partition)
                   for name in FEATURES}
        combined[partition] = {'trips': len(group), 'pct_at_1_second': float(np.mean(dt == 1)),
                               'median_track_km': float(group.track_km.median()),
                               'largest_gap_s': float(dt.max()), 'missing_feature_values': missing, **sums}
    combined['training_reference'] = {
        key: sum(row.get(key, 0) for row in read_json(processed / 'label_quality_report.json')['per_trip']
                 if row['partition'] == 'train')
        for key in ('weak_alignments', 'weak_candidate_misses', 'broken_route_connections', 'verified_labels')
    }
    write_json(combined, output / 'eda_summary.json')

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.05))
    for partition, color, label in (('train', GREEN, 'Training'), ('student_test', SAND, 'Student test')):
        group = audit[audit.partition == partition]
        dt = np.concatenate(interval_groups[partition])
        counts = [np.sum(dt <= 0), np.sum(dt == 1), np.sum((dt > 1) & (dt <= 5)),
                  np.sum((dt > 5) & (dt <= 30)), np.sum(dt > 30)]
        axes[0].plot(range(5), np.maximum(.1, counts), 'o-', label=label, color=color)
        axes[1].scatter(group.gps_rows, group.arc_count, label=label, color=color, s=15, alpha=.7)
        axes[2].scatter((group.west + group.east) / 2, (group.south + group.north) / 2, color=color,
                        s=15, alpha=.7, label=label)
    axes[0].set(yscale='log', ylabel='GPS intervals (log)', title='Sampling cadence', xticks=list(range(5)),
                xticklabels=['≤0', '1', '2–5', '6–30', '>30'], xlabel='Seconds')
    axes[1].set(title='Trip-network sizes', xscale='log', yscale='log', xlabel='GPS observations', ylabel='Directed arcs')
    axes[2].set(title='Trajectory centers', xlabel='Longitude', ylabel='Latitude')
    axes[2].legend(frameon=False, loc='lower right', fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.tight_layout(w_pad=2)
    save(fig, 'dataset_eda', plots)


def model_plots(processed, output, plots):
    comparison = read_json(output / 'model_comparison.json')
    models = comparison['models']
    labels = MODEL_LABELS[:-1] + [models[-1]['name']]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.1), sharey=True)
    selections = [('CV macro F1', [m['cv']['mean']['macro_f1'] for m in models]),
                  ('Student top-1', [m['student']['top1'] for m in models]),
                  ('Student macro F1', [m['student']['macro_f1'] for m in models]),
                  ('Student log loss', [m['student']['log_loss'] for m in models])]
    for ax, (title, values) in zip(axes, selections):
        ax.barh(labels, values, color=[TEAL] * 5 + [GREEN])
        ax.set_title(title, fontsize=10, loc='left', fontweight='bold')
        ax.set_xlim(0, 1.05 * max(values))
        ax.grid(axis='x', alpha=.18)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, 'model_comparison', plots)

    fig, axes = plt.subplots(2, 3, figsize=(9.1, 4.2))
    for ax, model in zip(axes.ravel(), models):
        counts = np.asarray(model['student']['confusion_matrix'])
        ax.imshow(counts / counts.sum(axis=1, keepdims=True), cmap='Greens', vmin=0, vmax=1)
        for i, j in np.ndindex(2, 2):
            ax.text(j, i, f'{counts[i, j]:,}', ha='center', va='center', fontsize=10,
                    color='white' if i == j else INK)
        ax.set(title=model['name'], xticks=[0, 1], yticks=[0, 1], xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    fig.tight_layout()
    save(fig, 'student_confusions', plots)

    bayes = read_json(output / 'bayesian_trials.json')
    scores = [trial['value'] for trial in bayes['trials']]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    x = np.arange(1, len(scores) + 1)
    ax.plot(x, scores, 'o', color=TEAL, label='Observed CV macro F1')
    ax.step(x, np.maximum.accumulate(scores), where='post', color=GREEN, label='Best to date')
    ax.axvspan(.5, 4.5, color='#e8f0e5', label='TPE startup trials')
    ax.set(xlabel='Trial', ylabel='Grouped CV macro F1', xticks=x)
    ax.ticklabel_format(style='plain', useOffset=False, axis='y')
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    save(fig, 'bayesian_trials', plots)

    training = load_training(processed)
    learn, valid = next(GroupKFold(n_splits=3).split(training, training.weak_target, groups=training.trip_id))
    train, validation = training.iloc[learn], training.iloc[valid]
    history_records = []
    fig, ax = plt.subplots(figsize=(8, 2.7))
    for record in [m for m in models if m['key'] == 'xgboost' or (m['key'] == 'tuned' and comparison['winner'] == 'xgboost')]:
        clf = estimator('xgboost', record['params'])
        clf.fit(train[FEATURES], train.weak_target, verbose=False,
                eval_set=[(train[FEATURES], train.weak_target), (validation[FEATURES], validation.weak_target)])
        history = clf.evals_result()
        for subset, style, values in (('fit', '--', history['validation_0']['logloss']),
                                      ('validation', '-', history['validation_1']['logloss'])):
            ax.plot(np.arange(len(values)) + 1, values, linestyle=style,
                    color=TEAL if record['key'] == 'xgboost' else GREEN, label=f'{record["name"]} {subset}')
            history_records.extend({'model': record['key'], 'subset': subset, 'iteration': index + 1,
                                    'log_loss': value} for index, value in enumerate(values))
    ax.set(xlabel='Boosting iteration', ylabel='Log loss', yscale='log')
    ax.legend(frameon=False, fontsize=8)
    save(fig, 'xgboost_history', plots)
    pd.DataFrame(history_records).to_csv(output / 'xgboost_fold1_history.csv', index=False)

    routes = read_json(output / 'route_comparison.json')['per_trip']
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 2.7))
    for ax, field, title in zip(axes, ['sequence_f1', 'continuity'], ['Ordered arc F1', 'Connected-pair continuity']):
        x = np.arange(len(routes['tuned']))
        ax.bar(x - .18, [row[field] for row in routes['tuned']], .36, color=TEAL, label='Classifier')
        ax.bar(x + .18, [row[field] for row in routes['hmm']], .36, color=GREEN, label='+ HMM')
        ax.set(title=title, ylim=(0, 1.05), xticks=x, xticklabels=[row['trip_id'][-2:] for row in routes['tuned']],
               xlabel='Student-test trip ID suffix')
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'route_comparison', plots)


def folium_map(raw, models, maps, plots, trip_id):
    path = raw / 'student_test' / trip_id
    matcher = MapMatcher(models)
    prepared, probabilities, hmm, baseline, _ = matcher.predict(path)
    payload = map_payload(prepared, probabilities, hmm, baseline)
    road = prepared['road']
    route = load_reference(path)
    edges, nodes = road['arc_raw'], road['nodes'][['longitude', 'latitude']].to_numpy()
    reference = []
    for arc in route:
        if 0 <= arc < len(edges) and road['usable'][arc]:
            reference.append(nodes[edges[arc]].tolist())
    (west, south), (east, north) = payload['bounds']
    figure = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles=None, control_scale=True,
                        prefer_canvas=True, zoom_control=True)
    # Layers use the supplied vector network. No third-party basemap is required.
    figure.default_js = [('leaflet', 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js')]
    figure.default_css = [('leaflet_css', 'https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css')]
    figure.get_root().header.add_child(Element('<style>.leaflet-container{background:#f0f2e9!important} '
        '.map-legend{position:fixed;bottom:19px;left:70px;z-index:9999;background:#fffefa;padding:10px 14px;'
        'border:1px solid #c4cdc4;border-radius:7px;color:#183b2c;font:12px system-ui;box-shadow:0 2px 6px #183b2c20}'
        '.map-legend b{display:block;margin-bottom:5px} .map-legend span{margin-right:12px;white-space:nowrap}</style>'))
    legend = f'<div class="map-legend"><b>Student-test trip {trip_id}</b><span style="color:#b95e38">● GPS</span>' \
             '<span style="color:#6f847a">━ Roads</span><span style="color:#4265a3">━ Reference</span>' \
             '<span style="color:#26805e">━ Predicted</span></div>'
    figure.get_root().html.add_child(Element(legend))

    def layer(name, lines, color, weight, opacity, dash=None, show=True):
        group = folium.FeatureGroup(name=name, show=show).add_to(figure)
        geojson = {'type': 'Feature', 'properties': {}, 'geometry': {
            'type': 'MultiLineString', 'coordinates': [line for line in lines if len(line) > 1]}}
        folium.GeoJson(geojson, style_function=lambda _: {'color': color, 'weight': weight, 'opacity': opacity,
                                                        'dashArray': dash}).add_to(group)

    network = payload['network']
    road_lines = [[network['vertices'][a], network['vertices'][b]] for a, b in network['edges']]
    layer('Road network', road_lines, '#829a8d', 1, .65)
    layer('Reference route', reference, '#4265a3', 6, .75)
    gps_layer = folium.FeatureGroup(name='Raw GPS').add_to(figure)
    for index, row in enumerate(payload['observations']):
        if row['raw'] is not None:
            folium.CircleMarker(row['raw'][::-1], radius=1.2, color=SAND, weight=0,
                                fill=True, fill_color=SAND, fill_opacity=.48, tooltip=f'GPS point {index + 1}').add_to(gps_layer)
    layer('Viterbi prediction', payload['route'], '#238362', 3, .95, '6 3')
    layer('Classifier only', payload['classifier'], '#a25491', 2, .9, '5 4', False)
    folium.LayerControl(collapsed=False).add_to(figure)
    figure.fit_bounds([[south, west], [north, east]], padding=(34, 30))
    figure.save(maps / f'student_trip_{trip_id}.html')

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    observations = prepared['observations']
    axes[0].scatter(observations.longitude, observations.latitude, s=2.5, c=observations.observation_index,
                    cmap='viridis', alpha=.8)
    axes[0].set(title=f'GPS path · Student test {trip_id}', xlabel='Longitude', ylabel='Latitude')
    axes[1].scatter(observations.timestamp_s, observations.speed_mps * 3.6, s=3, alpha=.55, color=GREEN)
    axes[1].set(title='GPS-derived speeds', xlabel='Raw timestamp (s)', ylabel='Speed (km/h)')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, 'gps_and_speeds', plots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--processed', type=Path, default=Path('data/processed'))
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--trip', default='00000007')
    parser.add_argument('--eda-only', action='store_true')
    args = parser.parse_args()
    plots, maps = args.results / 'plots', args.results / 'maps'
    for folder in (plots, maps):
        folder.mkdir(parents=True, exist_ok=True)
    explore(args.raw, args.processed, args.results, plots)
    if not args.eda_only:
        model_plots(args.processed, args.results, plots)
        folium_map(args.raw, args.models, maps, plots, args.trip)
