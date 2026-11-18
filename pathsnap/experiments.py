"""Train five candidate classifiers and tune the CV winner with Optuna TPE."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .evaluation import candidate_metrics, labeled_candidates, load_reference, reference_alignment
from .preprocessing import prepare_inference_trip
from .preprocessing.config import Config
from .preprocessing.geometry import FEATURES
from .preprocessing.io import discover, write_json
from .preprocessing.pipeline import BASE_KEYS

SEED = 42
MODELS = {
    'logistic': ('Logistic regression', 'Linear', LogisticRegression,
                 {'C': 1.0, 'max_iter': 300, 'random_state': SEED},
                 {'C': (0.03, 12.0, 'log')}),
    'gaussian_nb': ('Gaussian naive Bayes', 'Probabilistic', GaussianNB,
                     {'var_smoothing': 1e-5}, {'var_smoothing': (1e-9, 0.2, 'log')}),
    'tree': ('Decision tree', 'Tree', DecisionTreeClassifier,
             {'max_depth': 8, 'min_samples_leaf': 48, 'random_state': SEED},
             {'max_depth': (4, 16, 'int'), 'min_samples_leaf': (12, 120, 'int')}),
    'forest': ('Random forest', 'Bagged trees', RandomForestClassifier,
               {'n_estimators': 90, 'max_depth': 13, 'min_samples_leaf': 18,
                'max_features': 0.75, 'n_jobs': 4, 'random_state': SEED},
               {'n_estimators': (60, 140, 'int'), 'max_depth': (7, 19, 'int'),
                'min_samples_leaf': (8, 52, 'int'), 'max_features': (0.5, 1.0, 'float')}),
    'xgboost': ('XGBoost', 'Boosted trees', XGBClassifier,
                 {'n_estimators': 180, 'max_depth': 5, 'learning_rate': 0.06, 'subsample': 0.9,
                  'colsample_bytree': 0.9, 'min_child_weight': 5, 'reg_lambda': 2.0,
                  'tree_method': 'hist', 'objective': 'binary:logistic', 'eval_metric': 'logloss',
                  'n_jobs': 4, 'random_state': SEED},
                 {'n_estimators': (110, 260, 'int'), 'max_depth': (3, 8, 'int'),
                  'learning_rate': (0.025, 0.18, 'log'), 'min_child_weight': (3, 28, 'int'),
                  'reg_lambda': (0.5, 12.0, 'log')}),
}


def estimator(key, overrides):
    _, _, cls, defaults, _ = MODELS[key]
    model = cls(**(defaults | overrides))
    if key in ('logistic', 'gaussian_nb'):
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
    if key == 'xgboost':
        return model  # Native missing-value routing fitted on the training fold.
    return make_pipeline(SimpleImputer(strategy='median'), model)


def load_training(processed):
    features = pd.read_csv(processed / 'train_features.csv.gz', usecols=BASE_KEYS + FEATURES,
                           dtype={'trip_id': str})
    targets = pd.read_csv(processed / 'train_targets.csv.gz', usecols=BASE_KEYS + ['weak_target'],
                          dtype={'trip_id': str})
    if set(features.partition) != {'train'} or set(targets.partition) != {'train'}:
        raise ValueError('Training tables must contain only the training partition')
    result = targets.merge(features, on=BASE_KEYS, validate='one_to_one')
    if len(result) != len(targets):
        raise ValueError('Training features are missing target keys')
    return result.sort_values(BASE_KEYS, kind='stable', ignore_index=True)


def load_development(raw, config):
    labeled, all_candidates, counts = [], [], []
    for _, trip_id, path in discover(raw, 'student_test'):
        prepared = prepare_inference_trip(path, config)
        route = load_reference(path)
        if route is None:
            raise ValueError(f'Missing student reference for {trip_id}')
        alignment, quality = reference_alignment(trip_id, route, prepared, config)
        labeled.append(labeled_candidates(prepared, alignment))
        all_candidates.append(prepared['features'])
        counts.append({'trip_id': trip_id, 'observations': len(prepared['observations']),
                       'aligned': quality['weak_alignments'], 'misses': quality.get('weak_candidate_misses', 0)})
    return pd.concat(labeled, ignore_index=True), pd.concat(all_candidates, ignore_index=True), counts


def fit(model, rows):
    start = perf_counter()
    with threadpool_limits(limits=4):
        model.fit(rows[FEATURES], rows.weak_target)
    return perf_counter() - start


def score(model, rows):
    start = perf_counter()
    with threadpool_limits(limits=4):
        prediction = model.predict_proba(rows[FEATURES])[:, 1]
    return prediction, perf_counter() - start


def cross_validate(key, params, training, folds):
    details = []
    for fold_index, (learn_idx, valid_idx) in enumerate(folds, 1):
        learn, valid = training.iloc[learn_idx], training.iloc[valid_idx]
        model = estimator(key, params)
        train_s = fit(model, learn)
        probabilities, inference_s = score(model, valid)
        details.append({'fold': fold_index, 'fit_seconds': train_s, 'inference_seconds': inference_s,
                        **candidate_metrics(valid, probabilities)})
    average = {name: float(np.mean([fold[name] for fold in details]))
               for name in ('top1', 'macro_f1', 'log_loss', 'accuracy', 'macro_precision', 'macro_recall')}
    average['macro_f1_std'] = float(np.std([fold['macro_f1'] for fold in details]))
    return {'mean': average, 'folds': details}


def run(raw, processed, output, model_dir, config, trials):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    training = load_training(processed)
    development, all_candidates, coverage = load_development(raw, config)
    folds = list(GroupKFold(n_splits=3).split(training, training.weak_target, groups=training.trip_id))
    split = []
    for fold_index, (learn, valid) in enumerate(folds, 1):
        for role, indices in (('fit', learn), ('validation', valid)):
            split.extend({'fold': fold_index, 'role': role, 'trip_id': trip}
                         for trip in sorted(training.iloc[indices].trip_id.unique()))
    pd.DataFrame(split).to_csv(output / 'cv_splits.csv', index=False)
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(output / 'student_alignment_coverage.csv', index=False)
    records = []

    def compare(key, params, cv, tag):
        name, family, _, defaults, _ = MODELS[key]
        model = estimator(key, params)
        train_s = fit(model, training)
        train_p, _ = score(model, training)
        probabilities, inference_s = score(model, all_candidates)
        keys = all_candidates[BASE_KEYS].assign(probability=probabilities)
        labeled = development.merge(keys, on=BASE_KEYS, validate='one_to_one')
        artifact = model_dir / f'{tag}.joblib'
        joblib.dump(model, artifact, compress=3)
        record = {'key': tag, 'name': name + (' (tuned)' if tag == 'tuned' else ''),
                  'family': family, 'params': defaults | params, 'cv': cv,
                  'fit_seconds': train_s, 'student_inference_seconds': inference_s,
                  'student_scored_candidates': len(all_candidates),
                  'train': candidate_metrics(training, train_p),
                  'student': candidate_metrics(labeled, labeled.probability.to_numpy())}
        print(f'{tag:12} CV top-1={cv["mean"]["top1"]:.4f} macro-F1={cv["mean"]["macro_f1"]:.4f} '
              f'student top-1={record["student"]["top1"]:.4f} fit={train_s:.2f}s', flush=True)
        records.append(record)

    for key in MODELS:
        compare(key, {}, cross_validate(key, {}, training, folds), key)
    winner = max(records, key=lambda record: (record['cv']['mean']['top1'], record['cv']['mean']['macro_f1'],
                                             -record['cv']['mean']['log_loss']))['key']
    space = MODELS[winner][-1]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=4,
    ))

    def objective(trial):
        params = {}
        for name, (low, high, distribution) in space.items():
            if distribution == 'int':
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high, log=distribution == 'log')
        cv = cross_validate(winner, params, training, folds)
        trial.set_user_attr('cv', cv)
        print(f'Bayesian trial {trial.number + 1}/{trials}: macro-F1={cv["mean"]["macro_f1"]:.4f}', flush=True)
        return cv['mean']['macro_f1']

    tuning_start = perf_counter()
    study.optimize(objective, n_trials=trials)
    tuning_s = perf_counter() - tuning_start
    compare(winner, study.best_params, study.best_trial.user_attrs['cv'], 'tuned')
    write_json({
        'strategy': 'Optuna TPE', 'startup_trials': 4, 'seed': SEED, 'objective': 'mean grouped CV macro-F1',
        'winner': winner, 'space': space, 'best_trial': study.best_trial.number,
        'best_params': study.best_params, 'tuning_seconds': tuning_s,
        'trials': [{'number': t.number, 'params': t.params, 'value': t.value,
                    'seconds': t.duration.total_seconds(), 'cv': t.user_attrs['cv']} for t in study.trials],
    }, output / 'bayesian_trials.json')
    study.trials_dataframe(attrs=('number', 'value', 'duration', 'params', 'state')).to_csv(
        output / 'bayesian_trials.csv', index=False,
    )
    train_ids = {trip_id for _, trip_id, _ in discover(raw, 'train')}
    write_json({
        'seed': SEED, 'splits': 3, 'labels': 'Geometric proxy from chronological reference routes',
        'training_trips': len(train_ids), 'fitting_trips': training.trip_id.nunique(),
        'trips_without_proxy_targets': sorted(train_ids - set(training.trip_id)),
        'student_trips': len(coverage), 'student_observations': int(coverage_df.observations.sum()),
        'student_proxy_eligible': int((coverage_df.aligned + coverage_df.misses).sum()),
        'student_candidate_misses': int(coverage_df.misses.sum()),
        'selection_metric': 'grouped CV top-1, then macro-F1 and lower log loss',
        'winner': winner, 'wall_seconds': perf_counter() - started,
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__,
                    'optuna': optuna.__version__},
        'models': records,
    }, output / 'model_comparison.json')
    write_json({'model': records[-1]['name'], 'artifact': 'tuned.joblib', 'features': FEATURES,
                'training_family': winner, 'preprocessing': config.as_json()}, model_dir / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--processed', type=Path, default=Path('data/processed'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--config', type=Path, default=Path('configs/default.json'))
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    if args.trials < 5:
        parser.error('At least five trials are needed for TPE after four initial trials')
    run(args.raw, args.processed, args.out, args.models, Config.from_path(args.config), args.trials)
