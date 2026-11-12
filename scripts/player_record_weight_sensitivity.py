# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""One-parameter robustness check; see README.md and config/analysis_protocol.json.

Run from the repository: python scripts/player_record_weight_sensitivity.py
Existing predictions, source data, and manuscript files are read-only inputs.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.history_depth_analysis import (
    month_draws, tournament_draws, paired_interval, probability_metrics, write_csv,
)
from scripts.archetype_pair import fit_matchup_rate
from scripts.player_records import fit_player_skill_pooled, logit
from scripts.playerskill_general_only import K_O, K_S_INF, K_P
from scripts.unified_data import load_matches, load_holdout, _resolve, MATCHES, HOLDOUT

GRID = (0., .5, .75, 1., 1.25, 1.5)
VALIDATION = ('2024-04', '2024-05', '2024-06')
START, CUTOFF = '2024-07', '2026-10'
COLUMNS = ['match_id', 'tournament_id', 'month', 'player_A', 'player_B',
           'archetype_A', 'archetype_B', 'y']
PROC = RESULTS
PROTOCOL = PROTOCOL
SOURCES = [
    Path(__file__), PROTOCOL, ROOT / 'tests/test_player_record_weight_sensitivity.py',
    *[ROOT / 'scripts' / name for name in (
        'history_depth_analysis.py', 'archetype_pair.py',
        'player_records.py', 'playerskill_general_only.py',
        'logistic_baseline.py', 'unified_data.py', 'project_paths.py')],
]
REFERENCES = [PROC / name for name in (
    'FINAL_cardonly_vs_playerskill_predictions.csv',
    'FINAL_playerskill_generalonly_predictions.csv',
    'FINAL_playerskill_pooled_predictions.csv', 'FINAL_joint_predictions.csv',
    'FINAL_october_predictions.csv',
)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label(weight):
    return f'lambda={weight:g}'


def forecast_components(train, targets, cutoff):
    """Return fixed base log odds and player adjustment; never read target outcomes."""
    if cutoff > CUTOFF or any(r['month'] >= cutoff for r in train):
        raise ValueError('Training includes the forecast period or October outcomes')
    if any(r['month'] < cutoff or r['month'] > CUTOFF for r in targets):
        raise ValueError('Target month is outside the permitted forecast window')
    card = fit_matchup_rate(train, K_P)
    record = fit_player_skill_pooled(train, K_O, K_S_INF)
    z = np.empty((len(targets), 2))
    for i, r in enumerate(targets):
        z[i, 0] = logit(card(r['archetype_A'], r['archetype_B']))
        z[i, 1] = (record(r['player_A'], r['archetype_A'])
                   - record(r['player_B'], r['archetype_B']))
    return z


def probabilities(components, weights=GRID):
    z = components[:, :1] + components[:, 1:] * np.asarray(weights)
    # Stable sigmoid without changing the underlying formula.
    return np.exp(-np.logaddexp(0., -z))


def losses(prob, y):
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    y = np.asarray(y)[:, None]
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def select_weight(history):
    train = [r for r in history if r['month'] < VALIDATION[0]]
    val = [r for r in history if r['month'] in VALIDATION]
    if not train or set(r['month'] for r in val) != set(VALIDATION):
        raise ValueError('Incomplete training or validation period')
    z = forecast_components(train, val, VALIDATION[0])
    prob = probabilities(z)
    loss = losses(prob, [r['y'] for r in val])
    tids, tc = np.unique([r['tournament_id'] for r in val], return_inverse=True)
    counts = np.bincount(tc)
    scores = [dict(weight=w, match_log_loss=float(loss[:, j].mean()),
                   tournament_log_loss=float((np.bincount(tc, weights=loss[:, j]) / counts).mean()))
              for j, w in enumerate(GRID)]
    selected = min(scores, key=lambda s: (s['match_log_loss'], abs(s['weight'] - 1), s['weight']))
    info = dict(weight=selected['weight'], criterion='match-weighted validation log loss',
                grid=list(GRID), train_before='2024-04-01', training_matches=len(train),
                validation_months=list(VALIDATION), validation_matches=len(val),
                validation_tournaments=len(tids), validation_scores=scores,
                selection_uncertainty_in_intervals=False)
    return info, val, z


def historical_components(history, progress=False):
    if any(r['month'] >= CUTOFF for r in history):
        raise ValueError('October outcomes cannot enter expanding historical fits')
    by_month = defaultdict(list)
    for r in history:
        by_month[r['month']].append(r)
    train, targets, blocks, audit = [], [], [], []
    for month in sorted(by_month):
        batch = by_month[month]
        if month >= START:
            blocks.append(forecast_components(train, batch, month))
            targets.extend(batch)
            audit.append(dict(month=month, history_before=month+'-01',
                              training_matches=len(train), forecast_matches=len(batch),
                              max_training_month=max(r['month'] for r in train)))
            if progress:
                print(f'{month}: {len(train):,} training, {len(batch):,} forecasts', flush=True)
        train.extend(batch)
    if not blocks:
        raise ValueError('No historical evaluation matches')
    return targets, np.concatenate(blocks), audit


def write_predictions(path, rows, z):
    prob = probabilities(z)
    def records():
        for i, r in enumerate(rows):
            yield dict(match_id=r['match_id'], tournament_id=r['tournament_id'],
                       month=r['month'], history_before=min(r['month'], CUTOFF)+'-01',
                       observed_y=r['y'], pair_log_odds=float(z[i, 0]),
                       player_log_odds_difference=float(z[i, 1]),
                       **{label(w): float(prob[i, j]) for j, w in enumerate(GRID)})
    write_csv(path, records())


def read_references(rows, prob, loss, october=False):
    """Align existing predictions and audit the zero/one identities on every row."""
    index = {r['match_id']: i for i, r in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError('Duplicate match identity')
    context, audits = {}, {}
    specs = [
        ('pair', REFERENCES[0], 'cardonly_prob', 'cardonly_logloss', 0),
        ('overall', REFERENCES[1], 'prob', 'logloss', GRID.index(1.)),
        ('Archetype-specific records (unchanged)', REFERENCES[2], 'pooled_prob', 'pooled_logloss', None),
        ('Joint model (unchanged)', REFERENCES[3], 'prob', 'logloss', None),
    ]
    if october:
        specs = [(name, REFERENCES[4], col, None, j) for name, col, j in [
            ('pair', 'Card-only', 0), ('overall', 'Overall-record control', GRID.index(1.)),
            ('Archetype-specific records (unchanged)', 'Player-Skill', None),
            ('Joint model (unchanged)', 'Joint model', None)]]
    for name, path, pc, lc, j in specs:
        p, ll, seen = np.empty(len(rows)), np.empty(len(rows)), np.zeros(len(rows), bool)
        with path.open() as stream:
            for r in csv.DictReader(stream):
                i = index[r['match_id']]
                if seen[i] or r['tournament_id'] != rows[i]['tournament_id']:
                    raise ValueError('Duplicate prediction or tournament mismatch')
                if int(r['observed_y']) != rows[i]['y']:
                    raise ValueError('Prediction outcome mismatch')
                month = r['tournament_date'][:7] if october else r['month']
                if month != rows[i]['month']:
                    raise ValueError('Prediction month mismatch')
                p[i], seen[i] = float(r[pc]), True
                if lc:
                    ll[i] = float(r[lc])
        if not seen.all():
            raise ValueError('Missing reference predictions')
        if not lc:
            ll = losses(p[:, None], [r['y'] for r in rows])[:, 0]
        if j is not None:
            error = float(np.abs(p - prob[:, j]).max())
            if error > 5.00001e-7:
                raise AssertionError(f'{name} forecast reconciliation failed: {error}')
            loss_error = float(np.abs(ll - loss[:, j]).max())
            if lc and loss_error > 5.00001e-7:
                raise AssertionError(f'{name} loss reconciliation failed: {loss_error}')
            audits[name] = dict(rows=len(rows), max_probability_error=error,
                                max_loss_error=loss_error,
                                mean_loss_difference=float((loss[:, j] - ll).mean()))
        else:
            context[name] = ll
    return context, audits


def summarize_period(name, rows, prob, loss, context, selected, out):
    tids, codes = np.unique([r['tournament_id'] for r in rows], return_inverse=True)
    months = np.array([r['month'] for r in rows])
    y = np.array([r['y'] for r in rows])
    tmonths = np.empty(len(tids), dtype='U7')
    tmonths[codes] = months
    assert np.array_equal(tmonths[codes], months)
    counts = np.bincount(codes)
    t_losses = {label(w): np.bincount(codes, weights=loss[:, j]) / counts
                for j, w in enumerate(GRID)}
    scores = {}
    for j, w in enumerate(GRID):
        scores[label(w)] = dict(weight=w, selected=w == selected,
            tournament_log_loss=float(t_losses[label(w)].mean()),
            match_log_loss=float(loss[:, j].mean()),
            mean_absolute_distance_from_half=float(np.abs(prob[:, j] - .5).mean()),
            **probability_metrics(prob[:, j], y))
    for model, values in context.items():
        t_losses[model] = np.bincount(codes, weights=values) / counts
    calendar = sorted(set(months))
    if name == 'october':
        draws, units = tournament_draws(len(tids)), np.arange(len(tids))
        bootstrap_units = tids.tolist()
    else:
        draws = month_draws(len(calendar))
        units = np.array([calendar.index(m) for m in tmonths])
        bootstrap_units = calendar
    np.save(out / f'{name}_bootstrap_multiplicities.npy', draws)
    pairs = [(label(w), label(0.)) for w in GRID if w != 0.]
    pairs += [(label(selected), label(1.))]
    pairs += [(label(selected), model) for model in context]
    comparisons = {}
    for model, reference in dict.fromkeys(pairs):
        delta = t_losses[reference] - t_losses[model]
        comparisons[model+' vs '+reference] = dict(model=model, reference=reference,
            **paired_interval(delta, units, len(bootstrap_units), draws))
    write_csv(out / f'{name}_tournament_losses.csv', (
        dict(tournament_id=str(t), month=str(tmonths[i]), matches=int(counts[i]),
             **{model: float(values[i]) for model, values in t_losses.items()})
        for i, t in enumerate(tids)))
    return dict(matches=len(rows), tournaments=len(tids), months=calendar, scores=scores,
                comparisons=comparisons, bootstrap_units=bootstrap_units,
                unchanged_context_scores={m: float(t_losses[m].mean()) for m in context})


def report(summary):
    selection = summary['selection']
    chosen = label(selection['weight'])
    lines = ['# Player-record weight sensitivity results', '',
        f"Validation selected **{chosen}** by match-weighted log loss. Training ended before April 2024; April–June forecasts shared that fixed training set.", '',
        'This is a sensitivity analysis of the existing overall-record implementation. All other settings and information cutoffs remain fixed. No historical evaluation or October outcomes selected the coefficient.', '',
        '| Weight | Validation match-weighted log loss |', '|---:|---:|']
    for s in selection['validation_scores']:
        lines.append(f"| {s['weight']:g} | {s['match_log_loss']:.9f} |")
    lines += ['', '## Evaluation scores', '',
              '| Weight | Historical tournament log loss | Later historical | October |',
              '|---:|---:|---:|---:|']
    for w in GRID:
        values = [summary['periods'][p]['scores'][label(w)]['tournament_log_loss']
                  for p in ('primary', 'later', 'october')]
        lines.append(f'| {w:g} | '+' | '.join(f'{v:.9f}' for v in values)+' |')
    lines += ['', 'Positive paired gains favor the validation-selected variant. Intervals are pointwise and conditional on selection; October intervals are indicative.', '',
              '| Period | Selected variant versus | Gain [95% interval] |', '|---|---|---:|']
    for period, data in summary['periods'].items():
        for c in data['comparisons'].values():
            if c['model'] == chosen:
                lo, hi = c['ci']
                lines.append(f"| {period} | {c['reference']} | {c['gain']:.9f} [{lo:.9f}, {hi:.9f}] |")
    lines += ['', '## Supporting checks', '',
        '| Period | Weight | Match log loss | Brier score | Calibration RMSE | Mean distance from 0.5 |',
        '|---|---:|---:|---:|---:|---:|']
    for period, data in summary['periods'].items():
        for w in dict.fromkeys((0., 1., selection['weight'])):
            s = data['scores'][label(w)]
            lines.append(f"| {period} | {w:g} | {s['match_log_loss']:.9f} | {s['brier']:.9f} | {s['calibration_rmse']:.9f} | {s['mean_absolute_distance_from_half']:.9f} |")
    lines += ['', '## Interpretation limits', '',
        '- The later historical window overlaps the primary period.',
        '- The grid is descriptive outside validation; its best historical or October setting is not a newly selected model.',
        '- Archetype-specific and joint forecasts are unchanged contextual comparators, not equally retuned competitors.',
        '- Lambda weights predictive records; it does not measure intrinsic skill or a causal effect of experience.',
        '- These intervals omit coefficient-selection uncertainty and shared-model uncertainty. The single October period is not independent prospective confirmation.',
        '- New scores use full precision. Baseline forecasts reconcile with saved six-decimal predictions. Direct bootstrap differences may differ in the last endpoint digit from older sign-reversed intervals.',
        '', 'The manuscript and its original results have not been overwritten.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROC / 'player_record_weight')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    external(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inputs = [_resolve(MATCHES), _resolve(HOLDOUT), *REFERENCES]
    input_hashes = {artifact_label(p): digest(p) for p in inputs}
    source_hashes = {artifact_label(p): digest(p) for p in SOURCES}
    history = load_matches(COLUMNS)
    print(f'Loaded {len(history):,} pre-October records', flush=True)
    selection, val, vz = select_weight(history)
    (out / 'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    write_csv(out / 'validation_scores.csv', selection['validation_scores'])
    # Validation predictions share an April cutoff, not their forecast-month cutoff.
    vp = probabilities(vz)
    write_csv(out / 'validation_predictions.csv.gz', (
        dict(match_id=r['match_id'], tournament_id=r['tournament_id'], month=r['month'],
             history_before='2024-04-01', observed_y=r['y'],
             **{label(w): float(vp[i, j]) for j, w in enumerate(GRID)})
        for i, r in enumerate(val)))
    print(json.dumps({'selected_weight': selection['weight'],
                      'validation_scores': selection['validation_scores']}), flush=True)
    del val, vz, vp
    hist, hz, fit_audit = historical_components(history, progress=True)
    hold = load_holdout(COLUMNS)  # Selection is already written and frozen.
    oz = forecast_components(history, hold, CUTOFF)
    # Full-data outcome-exclusion check, not only a small synthetic test.
    without_outcomes = [{k: v for k, v in r.items() if k != 'y'} for r in hold]
    if not np.array_equal(oz, forecast_components(history, without_outcomes, CUTOFF)):
        raise AssertionError('October forecasts depend on October outcomes')
    assert len(hist) == 926811 and len(hold) == 48288
    assert not ({r['match_id'] for r in hist} & {r['match_id'] for r in hold})
    del history, without_outcomes
    periods, reconciliation = {}, {}
    for period, rows, z in [('primary', hist, hz), ('october', hold, oz)]:
        print(f'Scoring and reconciling {period}', flush=True)
        prob = probabilities(z)
        loss = losses(prob, [r['y'] for r in rows])
        context, reconciliation[period] = read_references(rows, prob, loss, period == 'october')
        write_predictions(out / f'{period}_predictions.csv.gz', rows, z)
        periods[period] = summarize_period(period, rows, prob, loss, context, selection['weight'], out)
        if period == 'primary':
            mask = np.array([r['month'] >= '2025-01' for r in rows])
            later = [r for r in rows if r['month'] >= '2025-01']
            periods['later'] = summarize_period('later', later, prob[mask], loss[mask],
                {m: v[mask] for m, v in context.items()}, selection['weight'], out)
    summary = dict(selection=selection, periods=periods, reconciliation=reconciliation,
                   smoothing=dict(k_p=K_P, k_o=K_O, k_s=K_S_INF), monthly_fits=fit_audit,
                   october_outcomes_removed_prediction_identity=True,
                   selection_recorded_before_evaluation=True,
                   full_precision_scoring=True, bootstrap_seed=20261011, bootstrap_draws=5000)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_csv(out / 'scores.csv', (dict(period=p, model=m, **{k:v for k,v in s.items() if k != 'bins'})
        for p, data in periods.items() for m, s in data['scores'].items()))
    write_csv(out / 'comparisons.csv', (dict(period=p, model=c['model'], reference=c['reference'],
        gain=c['gain'], lower=c['ci'][0], upper=c['ci'][1])
        for p, data in periods.items() for c in data['comparisons'].values()))
    (out / 'report.txt').write_text(report(summary))
    assert input_hashes == {artifact_label(p): digest(p) for p in inputs}
    assert source_hashes == {artifact_label(p): digest(p) for p in SOURCES}
    outputs = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic()-started, command=sys.argv,
                    python=sys.version, versions={n:importlib.metadata.version(n)
                        for n in ('numpy', 'scipy', 'scikit-learn')},
                    inputs=input_hashes, sources=source_hashes, outputs=outputs,
                    all_inputs_unchanged=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Completed in {manifest["elapsed_seconds"]:.1f}s: {out / "report.txt"}', flush=True)


if __name__ == '__main__':
    main()
