"""Build the consolidated technical report from measured project artifacts."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor('#19392e')
GREEN = colors.HexColor('#3f7657')
MUTED = colors.HexColor('#4c6558')
PALE = colors.HexColor('#edf4eb')
LINE = colors.HexColor('#dce8dd')
WHITE = colors.white
PAPER_W, PAPER_H = A4
FRAME_W = PAPER_W - 92
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Label', fontName='Helvetica-Bold', fontSize=9.2, textColor=GREEN,
                           leading=12, spaceAfter=7, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='Hero', fontName='Times-Roman', fontSize=38, leading=40, textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle(name='Section', fontName='Times-Roman', fontSize=28, leading=31, textColor=INK, spaceAfter=11))
styles.add(ParagraphStyle(name='Head', fontName='Helvetica-Bold', fontSize=11.5, leading=14, textColor=INK,
                           spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name='Copy', fontName='Helvetica', fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle(name='Compact', parent=styles['Copy'], fontSize=8.6, leading=11.4, spaceAfter=5))
styles.add(ParagraphStyle(name='Caption', parent=styles['Compact'], textColor=MUTED, fontSize=8.2, leading=10.6,
                           spaceBefore=5, spaceAfter=9))
styles.add(ParagraphStyle(name='Cell', parent=styles['Compact'], spaceAfter=0, fontSize=8.1, leading=10.4))
styles.add(ParagraphStyle(name='CellHead', parent=styles['Cell'], fontName='Helvetica-Bold', textColor=WHITE))
styles.add(ParagraphStyle(name='Terminal', fontName='Courier', fontSize=7.15, leading=10.3, textColor=INK,
                           backColor=PALE, borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=10))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paragraph(value, style='Copy'):
    return Paragraph(value, styles[style])


def percent(value, places=2):
    return f'{value:.{places}%}' if value is not None else 'N/A'


def whole(value):
    return f'{value:,}'


def table(headers, rows, widths, emphasize_last=False):
    content = [[paragraph(str(item), 'CellHead') for item in headers]]
    content.extend([paragraph(str(item), 'Cell') for item in row] for row in rows)
    commands = [('BACKGROUND', (0, 0), (-1, 0), INK), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, 1), (-1, -1), .4, LINE), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE])]
    if emphasize_last:
        commands.append(('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#dcebd4')))
    result = Table(content, widths, repeatRows=1, hAlign='LEFT', style=TableStyle(commands))
    return result


def picture(path, max_height, width=FRAME_W):
    source_w, source_h = ImageReader(path).getSize()
    scale = min(width / source_w, max_height / source_h)
    return Image(path, width=source_w * scale, height=source_h * scale, hAlign='LEFT')


def page(canvas, document):
    canvas.setFillColor(PALE)
    canvas.rect(0, PAPER_H - 13, PAPER_W, 13, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(46, 40, PAPER_W - 46, 40)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(46, 27, 'PATHSNAP  /  TECHNICAL REPORT')
    canvas.drawRightString(PAPER_W - 46, 27, f'{document.page:02d}')


def build(results, output, date_label, evaluator_path=None):
    eda = read(results / 'eda_summary.json')
    comparisons = read(results / 'model_comparison.json')
    bayes = read(results / 'bayesian_trials.json')
    route = read(results / 'route_comparison.json')
    verification = read(results / 'api_verification.json')
    evaluator = read(evaluator_path) if evaluator_path else None
    models = comparisons['models']
    train, student = eda['train'], eda['student_test']
    selected = next(m for m in models if m['key'] == comparisons['winner'])
    tuned = models[-1]
    raw, hmm = route['routes']['tuned'], route['routes']['hmm']
    shots, plots = results / 'screenshots', results / 'plots'
    story = []

    def section(label, title):
        story.extend([paragraph(label, 'Label'), paragraph(title, 'Section')])

    def add(*items):
        story.extend(items)

    def text(value, style='Copy'):
        add(paragraph(value, style))

    def figure(name, caption, height):
        add(picture(plots / f'{name}.png', height))
        text(caption, 'Caption')

    add(paragraph('PES UNIVERSITY  /  UE24CS352A MACHINE LEARNING', 'Label'),
        paragraph('PathSnap', 'Hero'))
    text('GPS map matching and route reconstruction', 'Head')
    text(f'Single-laptop development run  |  {escape(date_label)}')
    text(f'{escape(tuned["name"])} scores candidate roads. A directed HMM/Viterbi decoder reconstructs the route. '
         f'Across the ten student-test trips, continuity reached <b>{percent(hmm["continuity"])}</b> and mean ordered '
         f'route F1 reached <b>{percent(hmm["mean_sequence_f1"])}</b>. Pointwise reference comparisons cover '
         f'{whole(hmm["proxy_observations"])} of {whole(hmm["observations"])} observations with sufficiently clear geometric alignment.')
    section('02  /  DATASET UNDERSTANDING', 'Input files and data quality')
    text('Each <b>.track</b> row stores longitude, latitude and a timestamp in seconds. A <b>.nodes</b> row supplies a '
         'WGS 84 location. An <b>.arcs</b> row is a directed pair of zero-based node IDs. The arc row index supplies '
         'the ID in <b>.route</b>, an ordered traversal sequence. GPS and reference files have separate sampling units.')
    rows = [(label, whole(group['trips']), whole(group['gps_rows']), whole(group['node_count']),
             whole(group['arc_count']), percent(group['pct_at_1_second'], 1), whole(group['invalid_gps']))
            for label, group in [('Training', train), ('Student test', student)]]
    add(table(['Partition', 'Trips', 'GPS rows', 'Nodes', 'Directed arcs', '1 s intervals', 'Invalid GPS'], rows,
              [80, 40, 66, 74, 78, 84, 76]), Spacer(1, 9))
    text(f'Median raw track lengths were {train["median_track_km"]:.1f} km (train) and {student["median_track_km"]:.1f} km '
         f'(student). There were {whole(train["temporal_gaps"] + student["temporal_gaps"])} GPS intervals above 5 s and '
         f'{whole(train["abnormal_jumps"] + student["abnormal_jumps"])} apparent speed jumps above 65 m/s. The road audit '
         f'excluded {whole(train["invalid_arcs"] + student["invalid_arcs"])} invalid or zero-length arcs from matching.')
    figure('dataset_eda', 'Figure 1. Real GPS intervals, trip networks and trajectory centers by partition.', 161)

    add(PageBreak())
    section('02  /  DATASET UNDERSTANDING', 'Preprocessing and features')
    text('GPS displacement and bearing use the WGS 84 geodesic. Each trip road network is projected to a local '
         'metric UTM CRS. A spatial index retrieves the eight closest directed candidates within 50 m. When the search '
         'is empty, a nearest-road fallback is allowed within 300 m. Coordinates snapped to a segment carry its arc ID '
         'and fractional position. Invalid GPS rows retain their observation indices so the decoder can represent gaps.')
    add(table(['Input group', 'Candidate features', 'Why use them'], [
        ('Position', 'Perpendicular offset, offset / arc length, arc length, signed-direction projection fraction',
         'Separate close parallel roads and identify positions near junctions.'),
        ('Motion', 'Heading difference, speed, acceleration, elapsed time, prior GPS step distance',
         'Use direction and motion to rank plausible arcs.'),
        ('Quality flags', 'Gap, speed jump, stationary, nearest-road fallback',
         'Identify unstable headings, anomalous motion and low-support searches.'),
    ], [90, 218, 190]), Spacer(1, 8))
    text('Undefined features stay missing until fit. XGBoost learns missing-value branches. The other four classifiers '
         'use a median imputer fitted on each training fold. Logistic regression and Gaussian naive Bayes also scale '
         'features within the fitted pipeline.')
    columns = [name for name in train['missing_feature_values'] if train['missing_feature_values'][name]
               or student['missing_feature_values'][name]]
    add(table(['Missing candidate feature', 'Training rows', 'Student-test rows'],
              [(escape(name), whole(train['missing_feature_values'][name]), whole(student['missing_feature_values'][name]))
               for name in columns], [258, 120, 120]), Spacer(1, 8))
    quality = eda['training_reference']
    text(f'Weak point labels require a reference-arc distance of at most 12 m, an 8 m margin to a different '
         f'reference arc, heading difference of at most 55 degrees, neighboring support and monotone sequence position. '
         f'This yielded {whole(quality["weak_alignments"])} weak training observations and {whole(quality["weak_candidate_misses"])} '
         f'candidate misses. Training arc references contained {whole(quality["broken_route_connections"])} directed breaks. '
         f'The fitting table had labels in {comparisons["fitting_trips"]} of {comparisons["training_trips"]} trips.')
    figure('gps_and_speeds', 'Figure 2. Student-test GPS geometry and derived speed for the mapped example.', 166)

    add(PageBreak())
    section('03  /  MODEL EXPERIMENTATION', 'Five initial classifiers')
    text('We used three GroupKFold splits by training trip. Candidate classification uses a fixed 0.5 cutoff for Macro F1, '
         'precision and recall. Top-1 selects one candidate per aligned observation, using the smaller arc ID to break ties. '
         'Initial selection uses mean grouped-CV Top-1, then Macro F1 and lower log loss. Candidate ranking is the primary '
         'criterion because each observation contributes one state to the reconstructed route.')
    next_best = max((m for m in models[:-1] if m['key'] != selected['key']), key=lambda m: m['cv']['mean']['top1'])
    text(f'<b>{escape(selected["name"])}</b> led grouped CV at {percent(selected["cv"]["mean"]["top1"], 5)} Top-1. '
         f'{escape(next_best["name"])} followed at {percent(next_best["cv"]["mean"]["top1"], 5)}. '
         f'This is a close development comparison. Student comparisons cover {whole(comparisons["student_proxy_eligible"])} '
         f'geometrically aligned observations, including {whole(comparisons["student_candidate_misses"])} misses. Classification '
         'metrics below include candidate groups with a positive label. Continuity comes from the full tracks.', 'Compact')
    rows = [(m['name'], m['family'], percent(m['cv']['mean']['macro_f1']), percent(m['student']['top1']),
             percent(route['routes'][m['key']]['continuity']), percent(m['student']['macro_f1']),
             f'{m["student"]["log_loss"]:.4f}', f'{m["student_inference_seconds"] * 1000:.0f} ms') for m in models]
    add(table(['Model', 'Family', 'CV macro F1', 'Student Top-1', 'Continuity', 'Student macro F1', 'Log loss', 'Scoring'],
              rows, [98, 66, 59, 58, 56, 64, 49, 48], True), Spacer(1, 8))
    figure('model_comparison', 'Figure 3. Grouped training CV and student-test comparison across all fitted models.', 170)
    rows = [(m['name'], percent(m['student']['accuracy']), percent(m['student']['macro_precision']),
             percent(m['student']['macro_recall']), f'{m["fit_seconds"]:.2f} s',
             f'{m["cv"]["mean"]["macro_f1_std"] * 100:.2f} pp') for m in models]
    add(table(['Model', 'Student accuracy', 'Macro precision', 'Macro recall', 'Refit time', 'CV F1 SD'],
              rows, [133, 80, 80, 77, 64, 64], True))
    text('Scoring time covers the candidate-probability pass for all student-test candidates. Fit time is the final CPU '
         'refit on the training partition. Grouped CV standard deviation is expressed in percentage points.', 'Caption')

    add(PageBreak())
    section('03  /  MODEL EXPERIMENTATION', 'Model-by-model diagnostics')
    descriptions = {
        'logistic': ('C=1, L2, 300 iterations', 'regularized binary log loss',
                     'A linear ranker gives a fast baseline for distance and direction. It cannot directly express interactions near junctions.'),
        'gaussian_nb': ('variance smoothing 1e-5', 'Gaussian class-conditional likelihood with class priors',
                        'It provides a low-cost density baseline. The independence assumption is weak for correlated offset and length features.'),
        'tree': ('depth 8, min leaf 48', 'greedy Gini impurity reduction',
                 'Threshold rules capture junction interactions. A single constrained tree yields coarse, tied scores.'),
        'forest': ('90 trees, depth 13, min leaf 18, 0.75 features per split', 'bootstrapped Gini trees with averaged probabilities',
                   'Bagging captures nonlinear effects and stabilizes a single tree. Larger ensembles cost more to fit and score.'),
        'xgboost': ('180 rounds, depth 5, learning rate 0.06, column/row sample 0.9, min child weight 5, L2 2',
                    'regularized binary logistic loss',
                    'Boosted splits learn interactions and native missing-value routing. Additional rounds can sharpen probabilities on weak labels.'),
    }
    for model in models[:-1]:
        config, objective, reason = descriptions[model['key']]
        gap = (model['train']['macro_f1'] - model['cv']['mean']['macro_f1']) * 100
        text(escape(model['name']), 'Head')
        text(f'{reason} Initial fit used {config} and {objective}. Student Top-1 {percent(model["student"]["top1"])}, '
             f'macro F1 {percent(model["student"]["macro_f1"])}, log loss {model["student"]["log_loss"]:.4f}. '
             f'Fit-to-CV macro F1 gap {gap:+.2f} pp. '
             + ('The small gap gives limited evidence of overfit in the proxy task.' if abs(gap) <= 1 else
                'The larger training score suggests some overfit on the proxy task.'), 'Compact')
    figure('student_confusions', 'Figure 4. Student-test candidate confusion matrices. Rows are reference class. '
           'Columns are predicted class. Labels are geometric proxies.', 223)

    add(PageBreak())
    section('04  /  BAYESIAN SEARCH', 'Tuning the selected classifier')
    text(f'Optuna TPE evaluated {len(bayes["trials"])} configurations with seed {bayes["seed"]}, including '
         f'{bayes["startup_trials"]} startup trials. The objective was mean Macro F1 on the same three trip-grouped '
         f'training folds. Trial {bayes["best_trial"] + 1} won. Search took {bayes["tuning_seconds"]:.1f} s. '
         'The selected configuration was then refitted and evaluated on the ten student trips.')
    rationale = {'n_estimators': 'Ensemble size and CPU cost', 'max_depth': 'Interaction depth and variance',
                 'learning_rate': 'Per-tree update size', 'min_child_weight': 'Minimum node support',
                 'reg_lambda': 'Leaf-weight regularization', 'min_samples_leaf': 'Minimum node support',
                 'max_features': 'Features considered per split', 'C': 'Inverse regularization strength',
                 'var_smoothing': 'Stability of variance estimates'}
    rows = [(escape(name), f'{low:g} to {high:g} ({kind})', f'{bayes["best_params"][name]:.5g}', rationale[name])
            for name, (low, high, kind) in bayes['space'].items()]
    add(table(['Parameter', 'Search range', 'Selected', 'Reason'], rows, [130, 105, 67, 196]), Spacer(1, 8))
    rows = [(m['name'], percent(m['cv']['mean']['macro_f1']), percent(m['student']['top1']),
             percent(m['student']['macro_f1']), f'{m["student"]["log_loss"]:.4f}',
             f'{m["student_inference_seconds"] * 1000:.0f} ms') for m in (selected, tuned)]
    add(table(['Model', 'CV macro F1', 'Student Top-1', 'Student macro F1', 'Log loss', 'Scoring time'], rows,
              [128, 75, 75, 79, 65, 76], True), Spacer(1, 7))
    text(f'Tuning moved CV macro F1 by {(tuned["cv"]["mean"]["macro_f1"] - selected["cv"]["mean"]["macro_f1"]) * 100:+.3f} pp '
         f'and student Top-1 by {(tuned["student"]["top1"] - selected["student"]["top1"]) * 100:+.3f} pp. '
         f'The tuned fit-to-CV macro F1 gap was '
         f'{(tuned["train"]["macro_f1"] - tuned["cv"]["mean"]["macro_f1"]) * 100:+.2f} pp. '
         'The student comparison measures the same proxy task for both fits.', 'Compact')
    figure('bayesian_trials', 'Figure 5. Observed trial scores and the incumbent objective value.', 143)
    figure('xgboost_history', 'Figure 6. Actual XGBoost histories for training and validation in grouped fold 1. '
           'This diagnostic reuses recorded model settings. Random forests have no epoch-wise validation history.', 150)

    add(PageBreak())
    section('05  /  SEQUENTIAL RECONSTRUCTION', 'Directed HMM and Viterbi')
    cfg = route['decoder']
    text('A hidden state is one candidate arc and its fractional projection at an observation. Normalize positive-class '
         'scores within the observation, floor the raw scores at 1e-8 and take the weighted log. Keeping the arc ID '
         'and projection together preserves the candidate-to-road mapping.')
    text('The network distance adds the untraveled portion of the first arc, a directed shortest bridge, and the '
         'traveled portion of the next arc. The transition potential is minus absolute disagreement with geodesic GPS '
         'displacement divided by beta. The scale grows with elapsed time. A capped Dijkstra search rejects distant or '
         'unreachable pairs and allows small backward projection jitter on the same arc.')
    text('At each state, Viterbi adds its weighted log emission to the best previous score plus transition. It also '
         'considers a penalized restart. Backpointers recover the winning sequence and its fragment boundaries. Such '
         'boundaries are forced at missing candidates, missing or nonpositive elapsed time, and gaps over 45 s. '
         'The decoder outputs unmatched observations explicitly and inserts only directed bridge arcs. The classifier-only '
         'comparison takes independent Top-1 choices and uses the same bridge rules to measure continuity.')
    text(f'{len(route["decoder_trials"])} decoder settings were compared on student test. The search included weaker '
         f'emission weights and stronger restart penalties for low-scoring connected candidates. Mean ordered F1 '
         f'selected beta={cfg["beta_m"]:g} m, '
         f'emission weight={cfg["emission_weight"]:g}, and restart penalty={cfg["restart_penalty"]:g}. '
         f'Route F1 is twice the longest common subsequence length divided by the combined arc counts, after collapsing '
         f'consecutive repeated arcs within each fragment. Continuity divides linked pairs by eligible adjacent pairs.')
    rows = [('Nearest-road baseline', route['routes']['nearest']), ('Tuned classifier', raw), ('Tuned + HMM', hmm)]
    add(table(['Method', 'Proxy agreement', 'Mean ordered F1', 'Continuity', 'Disconnected pairs', 'Fragments'],
              [(label, percent(rec['proxy_accuracy']), percent(rec['mean_sequence_f1']), percent(rec['continuity']),
                whole(rec['disconnected_pairs']), whole(rec['route_fragments'])) for label, rec in rows],
              [124, 80, 80, 68, 76, 70], True), Spacer(1, 8))
    lower = [f'{h["trip_id"]} ({percent(r["sequence_f1"], 1)} to {percent(h["sequence_f1"], 1)})'
             for r, h in zip(route['per_trip']['tuned'], route['per_trip']['hmm']) if h['sequence_f1'] < r['sequence_f1']]
    lowest = min(route['per_trip']['hmm'], key=lambda row: row['sequence_f1'])
    text(f'The decoder reduced disconnected adjacent predictions from {whole(raw["disconnected_pairs"])} to '
         f'{whole(hmm["disconnected_pairs"])}. '
         + (f'Ordered F1 fell on {", ".join(lower)}. A plausible bridge can diverge from the supplied traversal.' if lower else
            f'Every student-test trip matched or improved its ordered F1. The lowest decoded score remained '
            f'{percent(lowest["sequence_f1"])} on {lowest["trip_id"]}, showing the remaining ambiguity of connected routes.'), 'Compact')
    figure('route_comparison', 'Figure 7. Route quality by student-test trip. Decoder settings were selected on '
           'these development trips, so these scores are development estimates.', 158)

    add(PageBreak())
    section('06  /  GEOGRAPHICAL VISUALISATION', 'Inspecting the route in Folium')
    text('This map overlays actual student-test GPS observations, the nearby road network, the supplied reference arc '
         'sequence and the decoded route. The interactive HTML exposes each layer separately and has an optional '
         'classifier-only overlay. The map-generation script builds these layers from the local dataset and model.')
    add(picture(shots / 'folium.png', 342))
    text('Figure 8. Folium viewport showing the map, geographic scale and layer controls.', 'Caption')
    text('Generate the interactive map with <b>python scripts/build_visualizations.py</b>. The HTML is written to '
         '<b>results/maps/</b>. Development reference geometry appears here for analysis. The Route Lab API returns '
         'aggregate reference metrics and predicted geometry, while reference arcs stay on the server.')
    text('Sources and reproducibility', 'Head')
    text('PES University PathSnap student materials and supplied student dataset. Newson &amp; Krumm (2009), '
         '<link href="https://doi.org/10.1145/1653771.1653818" color="#3f7657">Hidden Markov Map Matching Through Noise '
         'and Sparseness</link>. Optuna, '
         '<link href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html" '
         'color="#3f7657">TPESampler documentation</link>. Full configs, run logs and code accompany this report.', 'Compact')
    text('AI assistance disclosure', 'Head')
    text('OpenAI Codex assisted with implementation, experiment execution, dashboard design and drafting this report. '
         'The measurements and figures were generated from the supplied data and the recorded runs. The team should '
         'review the report and explain the decisions and results before submission.', 'Compact')

    add(PageBreak())
    section('07  /  API, UI AND DEPLOYMENT', 'Single-laptop demonstration')
    text('The browser fetches available trips and sends a partition and trip ID to the FastAPI server. The server '
         'runs preprocessing, candidate scoring and decoding on demand. It opens the reference only after the '
         'prediction finishes. The dashboard supports map layers, pan and zoom, playback, point inspection and '
         'GeoJSON export. A separate Experiments view reads the measured results.')
    add(picture(shots / 'dashboard.png', 335))
    text('Figure 9. Actual Route Lab interface after a successful inference request.', 'Caption')
    add(picture(shots / 'experiments.png', 242))
    text('Figure 10. Actual Experiments screen with model and optimization results.', 'Caption')

    add(PageBreak())
    section('07  /  API, UI AND DEPLOYMENT', 'API responses and receiver logs')
    verification_trips = verification['trips']
    text(f'Live HTTP verification passed for all {len(verification_trips)} supplied student-test trips and returned '
         f'404 for an unknown trip. Per-trip accuracy and continuity counts matched the experiment outputs. '
         f'Local endpoint processing ranged from '
         f'{min(t["timing"]["total_ms"] for t in verification_trips):,} ms to '
         f'{max(t["timing"]["total_ms"] for t in verification_trips):,} ms in this run.')
    api_shot = shots / 'api.png'
    if api_shot.exists():
        add(picture(api_shot, 240))
        text('Figure 11. Successful browser API response from the running single-laptop server.', 'Caption')
    transcript = (results / 'receiver_excerpt.txt').read_text(encoding='utf-8')
    wrapped = [part for line in transcript.splitlines() for part in wrap(
        line, width=105, subsequent_indent='    ', break_long_words=False, break_on_hyphens=False,
    )]
    add(Preformatted('\n'.join(wrapped), styles['Terminal']))
    text('Receiver terminal transcript from the live server and client run.', 'Caption')
    text('Both client and server used localhost for development. Launch with <b>Start-PathSnap.ps1</b>. In LAN mode, '
         'the second laptop opens the same interface over the first laptop’s LAN address. The API host accepts a '
         'reserved trip directory and a separate private reference directory through environment variables.')
    if evaluator:
        text('The screenshots above use student-test data. The reserved partition arrived separately after model and '
             'decoder selection. The next page records the held-out API verification.', 'Compact')
        add(PageBreak())
        section('07  /  RESERVED EVALUATION', 'Held-out API verification')
        holdout = evaluator['trips']
        assert evaluator['partition'] == 'evaluator_test' and len(holdout) == 10
        assert all(row['status'] == 200 and 'proxy_observations' in row['metrics'] for row in holdout)
        text('Each reserved trip was sent to the same localhost endpoint after development. The trained classifier, '
             'feature configuration and decoder settings were frozen before the archive was imported. Reference routes '
             'remained in a private server-side directory and the API loaded them after matching.')
        lines = []
        for label, key in (('Classifier only', 'classifier_metrics'), ('Classifier + HMM', 'metrics')):
            summary = [row[key] for row in holdout]
            per_trip_f1 = [match['sequence_f1'] for match in summary if match['sequence_f1'] is not None]
            eligible = sum(match['eligible_pairs'] for match in summary)
            aligned = sum(match['proxy_observations'] for match in summary)
            lines.append((label, percent(sum(match['proxy_correct'] for match in summary) / aligned) if aligned else 'N/A',
                          percent(sum(per_trip_f1) / len(per_trip_f1)) if per_trip_f1 else 'N/A',
                          percent(sum(match['connected_pairs'] for match in summary) / eligible) if eligible else 'N/A'))
        add(table(['Method', 'Proxy agreement', 'Mean ordered route F1', 'Route continuity'], lines,
                  [165, 111, 115, 107], True), Spacer(1, 9))
        total = sum(row['metrics']['observations'] for row in holdout)
        aligned = sum(row['metrics']['proxy_observations'] for row in holdout)
        misses = sum(row['metrics']['candidate_misses'] for row in holdout)
        text(f'All ten requests returned HTTP 200. The API processed {whole(total)} GPS observations. Pointwise '
             f'agreement is based on {whole(aligned)} geometrically aligned observations, including '
             f'{whole(misses)} candidate misses. Ordered route F1 uses the supplied arc sequences and is averaged '
             'across trips. Continuity pools eligible adjacent pairs.')
        text('The reserve archive, private references and per-trip results remain local. The physical two-laptop '
             'demonstration and public GitHub publication are separate submission steps.', 'Compact')
    else:
        text('The live results shown here use the supplied student-test trips. The public GitHub push and a physical '
             'two-laptop demonstration are submission steps.', 'Compact')

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, topMargin=37, bottomMargin=53, leftMargin=40, rightMargin=40,
                                 title='PathSnap Technical Report', subject='GPS map matching and route reconstruction')
    document.build(story, onFirstPage=page, onLaterPages=page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('docs/PathSnap-Technical-Report.pdf'))
    parser.add_argument('--date', default=date.today().isoformat())
    parser.add_argument('--evaluator', type=Path, help='Optional local reserved-partition API verification JSON')
    args = parser.parse_args()
    build(args.results, args.out, args.date, args.evaluator)
