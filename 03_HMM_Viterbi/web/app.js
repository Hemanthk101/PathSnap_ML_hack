import {RouteMap} from './route-map.js';

const $ = selector => document.querySelector(selector);
const fmt = new Intl.NumberFormat();
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escape = value => String(value).replace(/[&<>"']/g, character =>
  ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[character]));

const state = {partitions: [], partition: null, trip: null, result: null, request: null,
  index: 0, animation: 0, playStarted: null, elapsed: [], experiments: null, toast: null};
const map = new RouteMap($('#map'), index => { pause(); inspect(index); });

function time(value) {
  const seconds = Math.max(0, Math.floor(value));
  const hrs = Math.floor(seconds / 3600), mins = Math.floor(seconds / 60) % 60;
  return `${hrs ? `${hrs}:${String(mins).padStart(2, '0')}` : mins}:${String(seconds % 60).padStart(2, '0')}`;
}

async function api(path, options) {
  const response = await fetch(`/api/${path}`, options);
  if (!response.ok) {
    const issue = await response.json().catch(() => ({}));
    throw new Error(typeof issue.detail === 'string' ? issue.detail : `API request failed (${response.status}).`);
  }
  return response.json();
}

function status(label, variant) {
  $('#api-status').className = `status ${variant}`;
  $('#api-status span').textContent = label;
}

function mapMessage(title, message, error = false) {
  const element = $('#map-state');
  element.querySelector('strong').textContent = title;
  element.querySelector('p').textContent = message;
  element.classList.toggle('error', error);
  element.hidden = false;
}

function drawTrips() {
  const filter = $('#search').value.trim();
  const trips = (state.partition?.trips || []).filter(trip => trip.id.includes(filter));
  $('#trip-count').textContent = trips.length;
  const list = $('#trips');
  list.replaceChildren();
  for (const trip of trips) {
    const button = document.createElement('button');
    const selected = state.trip?.id === trip.id;
    button.className = selected ? 'trip-row active' : 'trip-row';
    button.setAttribute('aria-pressed', String(selected));
    const line = trip.observations == null ? 'Track unavailable' : `${fmt.format(trip.observations)} pts${
      trip.duration_s == null ? '' : ` · ${time(trip.duration_s)}`}`;
    button.innerHTML = `<span class="trip-glyph">${icon('route')}</span><span><strong>${escape(trip.id)}</strong><small>${line}</small></span>${icon('chevron')}`;
    button.addEventListener('click', () => {
      if (state.trip?.id === trip.id && (state.result || state.request)) return;
      state.trip = trip;
      drawTrips();
      const selectedRow = list.querySelector('.active');
      selectedRow?.focus({preventScroll: true});
      runMatch();
    });
    list.append(button);
  }
  if (!trips.length) {
    const message = document.createElement('p');
    message.className = 'list-empty';
    message.textContent = filter ? 'No trip IDs match your search.' : 'No trips are available.';
    list.append(message);
  }
}

function choosePartition(key) {
  state.partition = state.partitions.find(partition => partition.key === key);
  state.trip = state.partition?.trips[0];
  $('#search').value = '';
  $('#partition-hint').textContent = key === 'evaluator_test' ? 'Reserved evaluation. References stay on the API server.' :
    state.partition ? 'Development dataset. Select any trip to run live inference.' : 'Import the student dataset to enable local inference.';
  drawTrips();
  if (state.trip) runMatch();
}

async function initialize() {
  try {
    const catalog = await api('trips');
    status('API connected', 'ready');
    $('#model-name').textContent = `${catalog.model} + HMM`;
    state.partitions = catalog.partitions;
    const select = $('#partition');
    select.replaceChildren(...catalog.partitions.map(partition => new Option(partition.label, partition.key)));
    select.disabled = !catalog.partitions.length;
    choosePartition(select.value);
    if (!state.trip) mapMessage('No trips available', 'Import the student dataset, then reload the workspace.', true);
  } catch (error) {
    status('API unavailable', 'offline');
    mapMessage('Could not connect', error.message, true);
    $('#partition-hint').textContent = 'Check that the local API is running, then reload.';
  }
}

function clearReadouts() {
  for (const id of ['m-observations', 'm-continuity', 'm-agreement', 'm-time', 'detail-arc', 'detail-offset',
    'detail-score', 'detail-speed', 'base-cont', 'hmm-cont', 'base-f1', 'hmm-f1']) $(`#${id}`).textContent = '—';
  $('#s-observations').textContent = 'Track points';
  $('#s-continuity').textContent = 'Connected observation pairs';
  $('#s-agreement').textContent = 'Geometrically aligned points';
  $('#s-time').textContent = 'End-to-end API processing';
  $('#play-time').textContent = '0:00';
  $('#obs-counter').textContent = 'Observation —';
  $('#timeline').value = 0;
}

async function runMatch() {
  if (!state.trip) return;
  state.request?.abort();
  pause();
  const controller = new AbortController();
  state.request = controller;
  const id = state.trip.id, partition = state.partition.key;
  state.result = null;
  map.data = null;
  map.refresh();
  const badge = $('#active-trip');
  badge.textContent = id;
  badge.hidden = false;
  clearReadouts();
  for (const button of ['run', 'export', 'play', 'rewind', 'timeline']) $(`#${button}`).disabled = true;
  mapMessage('Reconstructing route', 'Finding candidate roads and decoding the most likely path.');
  $('#map-subtitle').textContent = `Processing trip ${id}`;
  try {
    const result = await api('match', {method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({trip_id: id, partition}), signal: controller.signal});
    if (state.request !== controller) return;
    state.result = result;
    showMatch(result);
    status('API connected', 'ready');
  } catch (error) {
    if (state.request !== controller || error.name === 'AbortError') return;
    mapMessage('Matching failed', error.message, true);
    $('#map-subtitle').textContent = 'Run matching to try again';
  } finally {
    if (state.request === controller) {
      state.request = null;
      $('#run').disabled = false;
    }
  }
}

function showMatch(result) {
  const {hmm, classifier} = result.metrics;
  const unit = (selector, value, suffix) => {
    const container = $(selector);
    const small = document.createElement('small');
    small.textContent = suffix;
    container.replaceChildren(document.createTextNode(value), small);
  };
  $('#m-observations').textContent = fmt.format(hmm.observations);
  $('#s-observations').textContent = `${fmt.format(hmm.matched)} matched to a road`;
  $('#m-continuity').textContent = percent(hmm.continuity);
  $('#s-continuity').textContent = `${fmt.format(hmm.connected_pairs)} / ${fmt.format(hmm.eligible_pairs)} linked pairs`;
  $('#m-agreement').textContent = percent(hmm.proxy_accuracy);
  $('#s-agreement').textContent = !result.reference_metrics_available ? 'Reference unavailable' :
    `${fmt.format(hmm.proxy_observations)} of ${fmt.format(hmm.observations)} points aligned`;
  const ms = result.timing.total_ms;
  unit('#m-time', ms < 1000 ? fmt.format(ms) : (ms / 1000).toFixed(2), ms < 1000 ? ' ms' : ' s');
  $('#s-time').textContent = `${fmt.format(result.timing.decoder_ms)} ms decoding`;
  $('#base-cont').textContent = percent(classifier.continuity);
  $('#hmm-cont').textContent = percent(hmm.continuity);
  $('#base-f1').textContent = percent(classifier.sequence_f1);
  $('#hmm-f1').textContent = percent(hmm.sequence_f1);
  $('#map-subtitle').textContent = `${hmm.route_km.toFixed(2)} km reconstructed · ${fmt.format(hmm.route_fragments)} ${hmm.route_fragments === 1 ? 'fragment' : 'fragments'}`;
  $('#map-state').hidden = true;
  map.setData(result.map);
  const obs = result.map.observations;
  const start = obs.find(point => point.time != null)?.time || 0;
  let last = 0;
  state.elapsed = obs.map(point => {
    if (point.time != null) last = Math.max(last, point.time - start);
    return last;
  });
  $('#timeline').max = Math.max(0, obs.length - 1);
  for (const button of ['export', 'play', 'rewind', 'timeline']) $(`#${button}`).disabled = !obs.length;
  inspect(0);
}

function inspect(index) {
  const point = state.result?.map.observations[index];
  if (!point) return;
  state.index = index;
  $('#timeline').value = index;
  $('#play-time').textContent = `${time(state.elapsed[index])} / ${time(state.elapsed.at(-1))}`;
  $('#obs-counter').textContent = `${fmt.format(index + 1)} / ${fmt.format(state.elapsed.length)} points`;
  $('#detail-arc').textContent = point.arc ?? 'Unmatched';
  $('#detail-offset').textContent = point.offset_m == null ? '—' : `${point.offset_m.toFixed(1)} m`;
  $('#detail-score').textContent = percent(point.candidate_score);
  $('#detail-speed').textContent = point.speed_mps == null ? '—' : `${(point.speed_mps * 3.6).toFixed(1)} km/h`;
  map.selected = index;
  map.drawSelection();
}

function pause() {
  cancelAnimationFrame(state.animation);
  state.animation = 0;
  state.playStarted = null;
  $('#play').innerHTML = icon('play');
  $('#play').title = 'Play trace';
  $('#play').setAttribute('aria-label', 'Play trace');
}

function togglePlayback() {
  if (state.animation) { pause(); return; }
  if (!state.elapsed.length || !state.result) return;
  if (state.index >= state.elapsed.length - 1) inspect(0);
  const origin = state.elapsed[state.index], speed = Number($('#speed').value);
  $('#play').innerHTML = icon('pause');
  $('#play').title = 'Pause trace';
  $('#play').setAttribute('aria-label', 'Pause trace');
  function advance(now) {
    state.playStarted ??= now;
    const elapsed = origin + (now - state.playStarted) * speed / 1000;
    let low = state.index, high = state.elapsed.length;
    while (low + 1 < high) {
      const mid = Math.floor((low + high) / 2);
      if (state.elapsed[mid] <= elapsed) low = mid;
      else high = mid;
    }
    if (low !== state.index) inspect(low);
    if (low >= state.elapsed.length - 1) pause();
    else state.animation = requestAnimationFrame(advance);
  }
  state.animation = requestAnimationFrame(advance);
}

function notify(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(state.toast);
  state.toast = setTimeout(() => { toast.hidden = true; }, 2600);
}

function exportRoute() {
  if (!state.result) return;
  const {trip_id, partition, map: geometry, metrics, model} = state.result;
  const features = geometry.route.filter(line => line.length >= 2).map((coordinates, index) => ({
    type: 'Feature', geometry: {type: 'LineString', coordinates},
    properties: {trip_id, partition, model, fragment: index + 1, route_continuity: metrics.hmm.continuity},
  }));
  for (const line of geometry.route.filter(points => points.length === 1)) {
    features.push({type: 'Feature', geometry: {type: 'Point', coordinates: line[0]},
      properties: {trip_id, partition, model, fragment: features.length + 1}});
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify({type: 'FeatureCollection', features}, null, 2)],
    {type: 'application/geo+json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = `pathsnap-${partition}-${trip_id}.geojson`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
  notify('Matched route exported');
}

function experimentView(data) {
  const modelRows = data.models.map(model => `<tr><td><strong>${escape(model.name)}</strong><small>${escape(model.family)}</small></td>
    <td>${percent(model.cv.mean.macro_f1)}</td><td>${percent(model.student.top1)}</td><td>${percent(model.student.macro_f1)}</td>
    <td>${model.student.log_loss.toFixed(3)}</td><td>${percent(data.routes[model.key].continuity)}</td>
    <td>${Math.round(model.student_inference_seconds * 1000)} ms</td></tr>`).join('');
  const bars = data.models.map(model => `<div class="bar-row"><span>${escape(model.name)}</span><span class="bar-track"><i style="width:${model.student.top1 * 100}%"></i></span><strong>${percent(model.student.top1)}</strong></div>`).join('');
  const min = Math.min(...data.trials), max = Math.max(...data.trials);
  const trials = data.trials.map((value, index) => `<button class="trial${index === data.best_trial ? ' best' : ''}" data-trial="${index}" style="height:${16 + 84 * (value - min) / Math.max(.00001, max - min)}%" title="Trial ${index + 1} · ${percent(value)}" aria-label="Trial ${index + 1}, ${percent(value)} macro F1"></button>`).join('');
  const routes = [['Pointwise proxy agreement', 'proxy_accuracy'], ['Ordered route F1', 'mean_sequence_f1'], ['Continuity', 'continuity']]
    .map(([title, key]) => `<div><span>${title}</span><strong>${percent(data.routes.tuned[key])}</strong><strong>${percent(data.routes.hmm[key])}</strong></div>`).join('');
  $('#experiment-content').innerHTML = `<div class="experiment-layout">
    <section class="surface experiment-panel"><h2>Model comparison</h2><p>Three training folds grouped by trip. Student-test metrics use geometrically aligned reference points.</p>
    <div class="experiment-table"><table><thead><tr><th>Model</th><th>CV macro F1</th><th>Student top-1</th><th>Student macro F1</th><th>Log loss</th><th>Continuity</th><th>Scoring time</th></tr></thead><tbody>${modelRows}</tbody></table></div></section>
    <div class="experiment-row"><section class="surface experiment-panel"><h2>Student-test top-1</h2><p>Highest-scoring candidate across ${data.student_trips} development trips.</p><div class="bar-list">${bars}</div></section>
    <section class="surface experiment-panel"><h2>Bayesian search</h2><p>Each trial optimizes grouped training CV macro F1. The chart is scaled to the observed trial range.</p><div class="trial-grid">${trials}</div><div class="trial-caption" id="trial-caption"></div></section></div>
    <section class="surface experiment-panel"><h2>Route reconstruction</h2><p>Independent classifier vs. the selected classifier with HMM/Viterbi.</p><div class="route-pairs"><div><span></span><strong>Classifier</strong><strong>+ HMM</strong></div>${routes}</div></section></div>`;
  function labelTrial(index) {
    $('#trial-caption').textContent = `Trial ${index + 1} · ${percent(data.trials[index])} CV macro F1${index === data.best_trial ? ' · selected' : ''}`;
  }
  labelTrial(data.best_trial);
  document.querySelectorAll('.trial').forEach(button => {
    const show = () => labelTrial(Number(button.dataset.trial));
    button.addEventListener('click', show);
    button.addEventListener('mouseenter', show);
    button.addEventListener('focus', show);
  });
}

async function display(section) {
  const showingLab = section === 'lab';
  $('#lab').hidden = !showingLab;
  $('#experiments').hidden = showingLab;
  for (const name of ['lab', 'experiments']) {
    $(`#nav-${name}`).classList.toggle('active', name === section);
    $(`#nav-${name}`).setAttribute('aria-pressed', String(name === section));
  }
  if (showingLab) return;
  pause();
  if (state.experiments) return;
  $('#experiment-content').textContent = 'Loading experiment results…';
  try {
    state.experiments = await api('experiments');
    experimentView(state.experiments);
  } catch (error) {
    $('#experiment-content').textContent = error.message;
  }
}

$('#nav-lab').addEventListener('click', () => display('lab'));
$('#nav-experiments').addEventListener('click', () => display('experiments'));
$('#partition').addEventListener('change', event => choosePartition(event.target.value));
$('#search').addEventListener('input', drawTrips);
$('#run').addEventListener('click', runMatch);
$('#export').addEventListener('click', exportRoute);
$('#play').addEventListener('click', togglePlayback);
$('#rewind').addEventListener('click', () => { pause(); inspect(0); });
$('#speed').addEventListener('change', () => { if (state.animation) { pause(); togglePlayback(); } });
$('#timeline').addEventListener('input', event => { pause(); inspect(Number(event.target.value)); });
$('#zoom-in').addEventListener('click', () => map.zoom(1.3));
$('#zoom-out').addEventListener('click', () => map.zoom(1 / 1.3));
$('#fit').addEventListener('click', () => map.fit());
for (const button of document.querySelectorAll('.layer')) button.addEventListener('click', () => {
  const layer = button.dataset.layer;
  map.layers[layer] = !map.layers[layer];
  button.classList.toggle('on', map.layers[layer]);
  button.setAttribute('aria-pressed', String(map.layers[layer]));
  map.refresh();
});
display('lab');
initialize();
