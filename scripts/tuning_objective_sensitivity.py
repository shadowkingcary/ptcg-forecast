# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Compare existing validation grids under match and tournament weighting.

Only the selection objective changes. See config/analysis_protocol.json.
All forecasts and intermediate arrays are written to the external workspace.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.project_paths import RESULTS, PROTOCOL, artifact_label, external
from scripts.archetype_pair import K_GRID, fit_matchup_rate
from scripts.player_records import POOLED_GRID, fit_player_skill_pooled, make_predict
from scripts.playerskill_general_only import K_S_INF
from scripts.joint_hierarchical_model import BASE, fit_predict as joint_predict
from scripts.match_data import prep, slicer
from scripts.unified_data import load_matches, _resolve, MATCHES, HOLDOUT
from scripts.verify_reproduction import check_inputs, compare, sha

VALIDATION = ('2024-04', '2024-05', '2024-06')
CUTOFF = '2026-10'
COLUMNS = ['match_id', 'tournament_id', 'month', 'player_A', 'player_B',
           'archetype_A', 'archetype_B', 'y']
OBJECTIVES = ('match', 'tournament')
MODELS = ('logistic', 'pair', 'overall', 'archetype_specific', 'joint')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_grid():
    return [BASE] + [tuple(b * (factor if i == j else 1.) for i, b in enumerate(BASE))
                     for j in range(3) for factor in (.5, 2.)]


def split_validation(history):
    train = [r for r in history if r['month'] < VALIDATION[0]]
    val = [r for r in history if r['month'] in VALIDATION]
    if not train or set(r['month'] for r in val) != set(VALIDATION):
        raise ValueError('Initial history or April-June validation is incomplete')
    return train, val


def loss_array(prob, outcomes):
    p = np.clip(np.asarray(prob, float), 1e-12, 1 - 1e-12)
    y = np.asarray(outcomes, float)
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def objective_scores(loss, tids):
    _, code = np.unique(tids, return_inverse=True)
    counts = np.bincount(code)
    return {'match_log_loss': float(np.mean(loss)),
            'tournament_log_loss': float((np.bincount(code, weights=loss) / counts).mean())}


def choose(candidates, objective):
    if objective not in OBJECTIVES:
        raise ValueError('Unknown selection objective')
    # Stable first-in-grid tie rule matches the original smoothing/scale loops.
    return min(candidates, key=lambda r: r[objective + '_log_loss'])


def forecast(train, targets, model, settings, cutoff):
    if cutoff > CUTOFF or any(r['month'] >= cutoff for r in train):
        raise ValueError('Training contains the forecast month or October outcomes')
    if any(r['month'] < cutoff or r['month'] > CUTOFF for r in targets):
        raise ValueError('Target is outside the forecast window')
    # Remove outcomes at the prediction boundary, including for joint encoding.
    clean = [{**r, 'y': 0} for r in targets]
    if model == 'joint':
        d = prep(train + clean)
        mask = np.arange(len(train) + len(clean)) < len(train)
        return joint_predict(slicer(d, mask), slicer(d, ~mask), d['n_ar'], settings['scales'])
    card = fit_matchup_rate(train, settings['k_p'])
    if model == 'pair':
        return np.array([card(r['archetype_A'], r['archetype_B']) for r in clean])
    if model not in ('overall', 'archetype_specific'):
        raise ValueError('Unknown model')
    record = fit_player_skill_pooled(train, settings['k_o'], settings['k_s'])
    predict = make_predict(card, record)
    return np.array([predict(r['player_A'], r['archetype_A'], r['player_B'], r['archetype_B'])
                     for r in clean])


def select_settings(history, prediction_sink=None):
    train, val = split_validation(history)
    y = np.array([r['y'] for r in val])
    tids = np.array([r['tournament_id'] for r in val])
    scores = []
    def candidate(model, settings):
        name = f'{model}_{sum(r["model"] == model for r in scores):02d}'
        p = forecast(train, val, model, settings, VALIDATION[0])
        row = {'candidate': name, 'model': model, 'settings': settings,
               **objective_scores(loss_array(p, y), tids)}
        scores.append(row)
        if prediction_sink is not None:
            prediction_sink[name] = p
        print(f'{name} {settings}: match={row["match_log_loss"]:.9f}, '
              f'tournament={row["tournament_log_loss"]:.9f}', flush=True)
        return row
    pairs = [candidate('pair', {'k_p': k}) for k in K_GRID]
    selected_pairs = {o: choose(pairs, o) for o in OBJECTIVES}
    pooled = {}
    # Preserve sequential selection: select pair smoothing, then freeze it while
    # choosing player smoothing; overall records inherit the selected k_o.
    for kp in dict.fromkeys(selected_pairs[o]['settings']['k_p'] for o in OBJECTIVES):
        pooled[kp] = [candidate('archetype_specific', {'k_p': kp, 'k_o': ko, 'k_s': ks})
                      for ko, ks in POOLED_GRID]
    joint = [candidate('joint', {'scales': list(sc)}) for sc in joint_grid()]
    settings = {}
    for objective in OBJECTIVES:
        pair = selected_pairs[objective]['settings']
        player = choose(pooled[pair['k_p']], objective)['settings']
        settings[objective] = {
            'logistic': {'C': 1.0}, 'pair': pair,
            'overall': {'k_p': pair['k_p'], 'k_o': player['k_o'], 'k_s': K_S_INF},
            'archetype_specific': player,
            'joint': choose(joint, objective)['settings'],
        }
    selection = {
        'training_before': '2024-04-01', 'training_matches': len(train),
        'validation_months': list(VALIDATION), 'validation_matches': len(val),
        'validation_tournaments': len(set(tids)), 'settings': settings,
        'changed_models': [m for m in MODELS if settings['match'][m] != settings['tournament'][m]],
        'selection_uses_only_initial_training_and_validation': True,
        'tie_rule': 'First candidate in the original grid order.',
    }
    return selection, scores, val


def original_settings(protocol):
    models = protocol['models']
    kp = models['pair']['selected_k']
    ko, ks = models['archetype_specific']['selected']
    return {
        'logistic': {'C': models['logistic']['C']}, 'pair': {'k_p': kp},
        'overall': {'k_p': kp, 'k_o': models['overall']['k_o'], 'k_s': models['overall']['k_s']},
        'archetype_specific': {'k_p': kp, 'k_o': ko, 'k_s': ks},
        'joint': {'scales': models['joint']['selected_scales']},
    }


def require_reusable_forecasts(selection, expected):
    """Never silently reuse baseline forecasts if either selection has changed."""
    if selection['settings']['match'] != expected:
        raise ValueError('Original match-weighted selection was not reproduced')
    changed = [m for m in MODELS if selection['settings']['tournament'][m] != expected[m]]
    if changed:
        raise ValueError('New evaluation forecasts are required for: ' + ', '.join(changed))
    if selection['changed_models']:
        raise ValueError('Inconsistent changed-model audit')


def reuse_evaluation(selection, protocol, expected):
    """Verify and reuse the existing forecasts for this observed no-change result.

    If a future implementation produces different selections, stop rather than
    presenting baseline scores as results for settings that have not been fit.
    """
    require_reusable_forecasts(selection, original_settings(protocol))
    ledger_path = RESULTS/'full_october_2026_run/paper_results.json'
    ledger = json.loads(ledger_path.read_text())
    source_hashes = {}
    for name, digest in ledger['sources'].items():
        source_hashes[name] = sha(RESULTS/name)
        if source_hashes[name] != digest:
            raise ValueError('Saved forecast file no longer matches the paper ledger: ' + name)
    errors = []
    for period in ('dataset', 'primary', 'later', 'october'):
        compare(expected[period], ledger[period], period, errors)
    if errors:
        raise ValueError('Original evaluation ledger differs from frozen reference: ' + str(errors[:5]))
    keys = {'logistic': 'M1', 'pair': 'Card-only', 'overall': 'Overall',
            'archetype_specific': 'Player-Skill', 'joint': 'Joint'}
    hold_keys = {**keys, 'overall': 'Overall-record control', 'joint': 'Joint model'}
    evaluation = []
    for period in ('primary', 'later', 'october'):
        for model in MODELS:
            values = ledger[period][(hold_keys if period == 'october' else keys)[model]]
            evaluation.append({'period': period, 'model': model,
                               'match_selected_tournament_log_loss': values['tournament_log_loss'],
                               'tournament_selected_tournament_log_loss': values['tournament_log_loss'],
                               'score_change': 0., 'match_weighted_log_loss': values['match_log_loss'],
                               'brier': values['brier'], 'calibration_rmse': values['calibration_rmse'],
                               'sharpness_mean_absolute_distance': values['mad']})
    return {
        'changed_models': [], 'refitted_evaluation_models': [],
        'reason': 'Both selection objectives reproduce the original settings for every main model.',
        'all_reported_forecasts_and_comparisons_unchanged': True,
        'not_an_independent_replication': True,
        'scope': 'Agreement within the original finite grids; not global optimality or evidence for untested grids.',
        'evaluation': evaluation,
        'historical_comparisons': ledger['comparisons'],
        'october_comparisons': ledger['october_comparisons'],
        'source_sha256': source_hashes,
        'paper_results_sha256': sha(ledger_path),
        'frozen_reference_verified': True,
    }


def report_text(selection, summary, scores):
    lines = ['Tuning-objective sensitivity check', '',
             'Training: fixed history before April 1, 2024.',
             f'Validation: April-June 2024; {selection["validation_matches"]:,} matches '
             f'in {selection["validation_tournaments"]:,} tournaments.',
             'No evaluation-period or October outcomes selected settings.', '',
             'Selected settings (match weighting / tournament weighting):']
    for model in MODELS:
        a, b = (selection['settings'][o][model] for o in OBJECTIVES)
        lines.append(f'  {model}: {a} / {b}')
    lines += ['', f'Candidate forecasts evaluated: {len(scores)}.',
              'The logistic penalty was fixed. Overall records inherited player smoothing.',
              'The player-adjustment coefficient remained one. Fitting objectives and update rules did not change.',
              '', 'No selected settings changed. No historical or October forecasts required refitting.',
              'The existing forecast hashes and all five models in all three scoring periods match the frozen paper reference.',
              'All reported scores, intervals, and conclusions are unchanged.', '',
              'Evaluation scores (identical under the two selections):',
              'period,model,tournament_log_loss,score_change']
    for row in summary['evaluation']:
        lines.append(f'{row["period"]},{row["model"]},'
                     f'{row["match_selected_tournament_log_loss"]:.9f},0')
    lines += ['', 'Interpretation limits:',
              'Agreement applies only to the original candidate grids, with stable first-in-grid tie handling.',
              'Some selected values lie at grid boundaries. This check does not establish globally optimal hyperparameters.',
              'The unchanged forecasts are reused evidence, not an independent validation or a new holdout.',
              'Existing confidence intervals remain conditional on the selected model settings.',
              '', 'See validation_scores.csv for every candidate and both objectives; selection.json for the exact settings;',
              'evaluation_scores.csv and summary.json for all evaluation scores and existing comparisons;',
              'manifest.json for input, code, output, and saved-forecast checksums.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=RESULTS/'tuning_objective')
    parser.add_argument('--publish-results', action='store_true',
                        help='Also copy compact numerical results into results/tuning_objective in the repository')
    args = parser.parse_args()
    if args.selection_only and args.publish_results:
        parser.error('Publish only a completed check, not selection-only outputs')
    expected = check_inputs()
    protocol = json.loads(PROTOCOL.read_text())
    experiment_protocol = protocol['tuning_objective_sensitivity']
    out = external(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    source_paths = [Path(__file__), PROTOCOL, ROOT/'tests/test_tuning_objective_sensitivity.py',
                    *[ROOT/'scripts'/name for name in ('archetype_pair.py', 'player_records.py',
                    'joint_hierarchical_model.py', 'match_data.py', 'logistic_baseline.py',
                    'unified_data.py', 'project_paths.py', 'playerskill_general_only.py', 'verify_reproduction.py')]]
    source_hashes = {artifact_label(p): sha(p) for p in source_paths}
    history = load_matches(COLUMNS)  # loader excludes every October row
    predictions = {}
    selection, scores, val = select_settings(history, predictions)
    # Persist selection before accessing holdout forecasts or scoring later periods.
    write_json(out/'selection.json', selection)
    selection_saved = datetime.now(timezone.utc).isoformat()
    write_csv(out/'validation_scores.csv', [{**r, 'settings': json.dumps(r['settings'], sort_keys=True)} for r in scores])
    np.savez_compressed(out/'validation_predictions.npz', **predictions)
    with gzip.open(out/'validation_rows.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['match_id', 'tournament_id', 'month', 'y'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in val)
    write_json(out/'validation_manifest.json', {
        'started_utc': start, 'selection_saved_utc': selection_saved,
        'input_sha256': {artifact_label(_resolve(n)): sha(_resolve(n)) for n in (MATCHES, HOLDOUT)},
        'protocol': experiment_protocol,
        'script_sha256': sha(Path(__file__)),
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
    })
    print(json.dumps(selection, indent=2), flush=True)
    if args.selection_only:
        return
    summary = reuse_evaluation(selection, protocol, expected)
    summary['selection'] = selection
    write_json(out/'summary.json', summary)
    write_csv(out/'evaluation_scores.csv', summary['evaluation'])
    (out/'report.txt').write_text(report_text(selection, summary, scores))
    outputs = ['selection.json', 'validation_scores.csv', 'evaluation_scores.csv', 'summary.json', 'report.txt']
    assert source_hashes == {artifact_label(p): sha(p) for p in source_paths}, 'Source changed during run'
    write_json(out/'manifest.json', {
        'started_utc': start, 'selection_saved_utc': selection_saved,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': {artifact_label(_resolve(n)): sha(_resolve(n)) for n in (MATCHES, HOLDOUT)},
        'source_sha256': source_hashes, 'protocol': experiment_protocol,
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
        'output_sha256': {p: sha(out/p) for p in outputs},
        'external_validation_audit_sha256': {p: sha(out/p) for p in ('validation_predictions.npz', 'validation_rows.csv.gz')},
        'evaluation_refits': 0, 'passed': True,
    })
    if args.publish_results:
        publication = ROOT/'results/tuning_objective'
        publication.mkdir(parents=True, exist_ok=True)
        for name in outputs + ['manifest.json']:
            shutil.copy2(out/name, publication/name)
    print('PASS: both objectives reproduce every original setting; forecasts and comparisons unchanged.')


if __name__ == '__main__':
    main()
