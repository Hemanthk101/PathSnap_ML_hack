"""PathSnap API and same-origin dashboard for local or LAN deployment."""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .evaluation import load_reference, reference_alignment, route_metrics
from .inference import MapMatcher, map_payload
from .preprocessing.io import DataError, read_numeric

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger('uvicorn.error')


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r'^\d{8}$')
    partition: str = Field(default='student_test', pattern=r'^(student_test|evaluator_test)$')


def trip_catalog(folder):
    catalog = {}
    if folder is not None and folder.is_dir():
        for path in sorted(folder.iterdir()):
            if not path.is_dir() or not re.fullmatch(r'[0-9]{8}', path.name):
                continue
            if all((path / f'{path.name}.{ext}').is_file() for ext in ('track', 'nodes', 'arcs')):
                catalog[path.name] = path
    return catalog


def create_app(student_root=None, evaluator_root=None, reference_root=None, model_dir=None):
    def optional_path(argument, key):
        value = argument if argument is not None else os.getenv(key)
        return Path(value) if value else None

    student = optional_path(student_root, 'PATHSNAP_STUDENT_ROOT') or ROOT / 'data/raw/student_test'
    evaluator = optional_path(evaluator_root, 'PATHSNAP_EVALUATOR_ROOT')
    references = optional_path(reference_root, 'PATHSNAP_EVALUATOR_REFERENCE_ROOT')
    model_dir = optional_path(model_dir, 'PATHSNAP_MODEL_DIR') or ROOT / 'models'
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(application):
        application.state.matcher = MapMatcher(model_dir)
        LOGGER.info('PathSnap ready | model=%s', application.state.matcher.manifest['model'])
        yield

    app = FastAPI(title='PathSnap', version='1.0.0', lifespan=lifespan)
    app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')

    @app.get('/', include_in_schema=False)
    def dashboard():
        return FileResponse(ROOT / 'web/index.html')

    @app.get('/api/health')
    def health():
        return {'status': 'ready', 'model': app.state.matcher.manifest['model']}

    @app.get('/api/trips')
    def trips():
        partitions = []
        for key, label, folder in (('evaluator_test', 'Evaluator trips', evaluator),
                                    ('student_test', 'Student-test trips', student)):
            catalog = trip_catalog(folder)
            if not catalog:
                continue
            records = []
            for trip_id, path in catalog.items():
                try:
                    data = read_numeric(path / f'{trip_id}.track', 3)
                    duration = float(data[-1, 2] - data[0, 2])
                except DataError:
                    records.append({'id': trip_id, 'observations': None, 'duration_s': None})
                    continue
                records.append({'id': trip_id, 'observations': len(data),
                                'duration_s': round(duration) if 0 <= duration < float('inf') else None})
            partitions.append({'key': key, 'label': label, 'trips': records})
        return {'partitions': partitions, 'model': app.state.matcher.manifest['model']}

    @app.get('/api/experiments')
    def experiments():
        output = ROOT / 'results'
        model = json.loads((output / 'model_comparison.json').read_text(encoding='utf-8'))
        route = json.loads((output / 'route_comparison.json').read_text(encoding='utf-8'))
        optimization = json.loads((output / 'bayesian_trials.json').read_text(encoding='utf-8'))
        return {'models': [{k: row[k] for k in ('key', 'name', 'family', 'cv', 'student', 'fit_seconds',
                                               'student_inference_seconds')} for row in model['models']],
                'routes': route['routes'], 'trials': [trial['value'] for trial in optimization['trials']],
                'best_trial': optimization['best_trial'], 'winner': model['winner'],
                'student_trips': model['student_trips']}

    @app.post('/api/match')
    def match(request: MatchRequest):
        selected_root = evaluator if request.partition == 'evaluator_test' else student
        path = trip_catalog(selected_root).get(request.trip_id)
        if path is None:
            raise HTTPException(status_code=404, detail='Trip is unavailable in the selected partition')
        start = perf_counter()
        matcher = app.state.matcher
        LOGGER.info('match started | partition=%s trip=%s', request.partition, request.trip_id)
        with inference_lock:
            try:
                prepared, probabilities, hmm, baseline, timing = matcher.predict(path)
            except DataError as exc:
                LOGGER.warning('match input invalid | trip=%s error=%s', request.trip_id, exc)
                raise HTTPException(status_code=422, detail='This trip contains invalid input data') from exc
            # Reference access begins only after both prediction outputs are finalized.
            reference_dir = references if request.partition == 'evaluator_test' else None
            route = None
            try:
                route = load_reference(path, reference_dir)
            except DataError as exc:
                LOGGER.warning('reference unavailable | trip=%s error=%s', request.trip_id, exc)
            alignment, _ = reference_alignment(request.trip_id, route, prepared, matcher.config)
            results = {'hmm': route_metrics(hmm, alignment, route),
                       'classifier': route_metrics(baseline, alignment, route)}
            geometry = map_payload(prepared, probabilities, hmm, baseline)
        timing['total_ms'] = round((perf_counter() - start) * 1000)
        LOGGER.info('match completed | partition=%s trip=%s observations=%s linked=%s proxy_correct=%s elapsed_ms=%s',
                    request.partition, request.trip_id, results['hmm']['observations'],
                    results['hmm']['connected_pairs'], results['hmm'].get('proxy_correct'), timing['total_ms'])
        return {'trip_id': request.trip_id, 'partition': request.partition, 'model': matcher.manifest['model'],
                'reference_metrics_available': route is not None, 'metrics': results,
                'timing': timing, 'map': geometry}

    return app


app = create_app()
