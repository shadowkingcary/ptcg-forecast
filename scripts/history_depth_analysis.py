# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Evaluate existing player-record forecasts by pre-prediction history depth.

Run from the repository: python scripts/history_depth_analysis.py
No model is fitted, tuned, or updated. See README.md and config/analysis_protocol.json.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import io
from itertools import groupby
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.unified_data import load_matches, load_holdout, _resolve, MATCHES, HOLDOUT

PROC = RESULTS
START, CUTOFF = '2024-07', '2026-10'
BINS = ['0', '1-9', '10-49', '50+']
MODELS = ['Archetype-pair', 'Overall records', 'Player-Skill']
CONTRASTS = [('Overall vs pair', 0, 1), ('Player-Skill vs pair', 0, 2), ('Player-Skill vs overall', 1, 2)]
SEED, REPLICATES, BLOCK = 20261011, 5000, 4
COLUMNS = ['match_id', 'tournament_id', 'tournament_date', 'month', 'player_A', 'player_B', 'archetype_A', 'archetype_B', 'y']
PREDICTIONS = [
    ('FINAL_cardonly_vs_playerskill_predictions.csv', 'cardonly_prob', 'cardonly_logloss'),
    ('FINAL_playerskill_generalonly_predictions.csv', 'prob', 'logloss'),
    ('FINAL_playerskill_pooled_predictions.csv', 'pooled_prob', 'pooled_logloss')]


def depth_bin(n):
    if n < 0:
        raise ValueError('Prior count must be nonnegative')
    return '0' if n == 0 else '1-9' if n < 10 else '10-49' if n < 50 else '50+'


def history_descriptors(history, holdout):
    """Count appearances only; outcome fields are never accessed here."""
    counts, specific, pairs = Counter(), Counter(), Counter()

    def describe(r):
        a, b, x, y = (r[k] for k in ('player_A', 'player_B', 'archetype_A', 'archetype_B'))
        na, nb = counts[a], counts[b]
        return {'match_id': r['match_id'], 'tournament_id': r['tournament_id'], 'month': r['month'],
                'history_before': min(r['month'], CUTOFF) + '-01',
                'prior_A': na, 'prior_B': nb, 'prior_A_archetype': specific[a, x],
                'prior_B_archetype': specific[b, y], 'prior_pair': pairs[tuple(sorted((x, y)))],
                'min_prior': min(na, nb), 'depth_group': depth_bin(min(na, nb)),
                'new_players': int(na == 0) + int(nb == 0)}

    for month, iterator in groupby(sorted(history, key=lambda r: r['month']), key=lambda r: r['month']):
        if month >= CUTOFF:
            raise ValueError('October must not enter the historical count updates')
        batch = list(iterator)
        if month >= START:
            for row in batch:
                yield describe(row)
        for row in batch:
            a, b, x, y = (row[k] for k in ('player_A', 'player_B', 'archetype_A', 'archetype_B'))
            counts[a] += 1; counts[b] += 1
            specific[a, x] += 1; specific[b, y] += 1
            pairs[tuple(sorted((x, y)))] += 1
    for row in holdout:
        if row['month'] != CUTOFF:
            raise ValueError('Holdout descriptor must refer to October 2026')
        yield describe(row)


def month_draws(n_months, replicates=REPLICATES, seed=SEED, block=BLOCK):
    if n_months < block:
        raise ValueError('Too few calendar months for the requested blocks')
    draws = np.zeros((replicates, n_months), dtype=np.int16)
    rng = random.Random(seed)
    for row in draws:
        for _ in range(math.ceil(n_months / block)):
            start = rng.randint(0, n_months - block)
            row[start:start + block] += 1
    return draws


def tournament_draws(n, replicates=REPLICATES):
    indices = np.random.default_rng(SEED).integers(0, n, (replicates, n))
    return np.array([np.bincount(row, minlength=n) for row in indices], dtype=np.int16)


def paired_interval(values, units, n_units, draws, reverse_endpoints=False):
    """Bootstrap a ratio over a FIXED calendar/event universe, including gaps."""
    values, units = np.asarray(values), np.asarray(units, dtype=int)
    if not len(values):
        return {'gain': None, 'ci': None, 'zero_support_replicates': len(draws)}
    totals = np.bincount(units, weights=values, minlength=n_units)
    counts = np.bincount(units, minlength=n_units)
    denom = draws @ counts
    bad = int(np.sum(denom == 0))
    interval = None
    if bad == 0:
        samples = (draws @ totals) / denom
        # The existing October ledger reverses the opposite-direction interval.
        # Discrete order statistics differ by one index under sign reversal.
        samples = np.sort(-samples if reverse_endpoints else samples)
        interval = [float(samples[int(.025 * len(samples))]), float(samples[int(.975 * len(samples))])]
        if reverse_endpoints:
            interval = [-interval[1], -interval[0]]
    return {'gain': float(np.mean(values)), 'ci': interval, 'zero_support_replicates': bad}


def tournament_means(loss, codes, n_tournaments):
    counts = np.bincount(codes, minlength=n_tournaments)
    used = counts > 0
    means = np.stack([np.bincount(codes, weights=loss[:, k], minlength=n_tournaments)[used] / counts[used]
                      for k in range(loss.shape[1])], axis=1)
    return np.flatnonzero(used), counts[used], means


def probability_metrics(prob, y):
    bins = np.minimum((prob * 10).astype(int), 9)
    n = np.bincount(bins, minlength=10)
    sp = np.bincount(bins, weights=prob, minlength=10)
    sy = np.bincount(bins, weights=y, minlength=10)
    used = n > 0
    return {'brier': float(np.mean((prob-y)**2)),
            'calibration_rmse': float(np.sqrt(np.sum((sp[used]-sy[used])**2/n[used])/len(prob))),
            'bins': [{'bin': int(b), 'matches': int(n[b]), 'mean_prediction': float(sp[b]/n[b]),
                      'observed': float(sy[b]/n[b])} for b in np.flatnonzero(used)]}


def write_csv(path, rows, fields=None):
    iterator = iter(rows)
    first = next(iterator, None)
    if fields is None:
        fields = list(first) if first else []
    # Reproducible gzip headers: no wall-clock timestamp or original filename.
    with path.open('wb') as raw:
        if path.suffix == '.gz':
            zipped = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0)
            text = io.TextIOWrapper(zipped, encoding='utf-8', newline='')
        else:
            text = io.TextIOWrapper(raw, encoding='utf-8', newline='')
        with text:
            writer = csv.DictWriter(text, fieldnames=fields); writer.writeheader()
            if first is not None: writer.writerow(first)
            writer.writerows(iterator)


def align_predictions(descriptors, expected):
    index = {r['match_id']: i for i, r in enumerate(descriptors)}
    if len(index) != len(descriptors) or set(index) != set(expected):
        raise ValueError('Descriptor identities do not match the eligible evaluation population')
    n = len(index)
    probs, losses = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    months = np.array([r['month'] for r in descriptors])
    historical = months < CUTOFF
    sources = []
    for k, (filename, pc, lc) in enumerate(PREDICTIONS):
        path = PROC / filename; sources.append(path); seen = set()
        with path.open() as stream:
            for r in csv.DictReader(stream):
                mid = r['match_id']; i = index[mid]
                if mid in seen or not historical[i]: raise ValueError('Duplicate or nonhistorical prediction')
                seen.add(mid)
                assert expected[mid] == (r['tournament_id'], r['month'], int(r['observed_y']))
                probs[i, k], losses[i, k] = float(r[pc]), float(r[lc])
        assert len(seen) == int(historical.sum())
    path = PROC / 'FINAL_october_predictions.csv'; sources.append(path); seen = set()
    with path.open() as stream:
        for r in csv.DictReader(stream):
            mid = r['match_id']; i = index[mid]
            if mid in seen or historical[i]: raise ValueError('Duplicate or non-October holdout forecast')
            seen.add(mid)
            y = int(r['observed_y'])
            assert expected[mid] == (r['tournament_id'], r['tournament_date'][:7], y)
            probs[i] = [float(r[x]) for x in ['Card-only', 'Overall-record control', 'Player-Skill']]
            p = probs[i]
            losses[i] = -(y*np.log(np.clip(p, 1e-12, 1))+(1-y)*np.log(np.clip(1-p, 1e-12, 1)))
    assert len(seen) == int((~historical).sum())
    assert np.isfinite(probs).all() and np.isfinite(losses).all()
    assert ((probs >= 0) & (probs <= 1)).all()
    return probs, losses, sources


def summarize_period(label, descriptors, probs, losses, expected, monthly_draws):
    months = np.array([r['month'] for r in descriptors])
    ids = [r['match_id'] for r in descriptors]
    y = np.array([expected[mid][2] for mid in ids])
    tids, codes = np.unique([r['tournament_id'] for r in descriptors], return_inverse=True)
    tmonths = np.empty(len(tids), dtype='U7'); tmonths[codes] = months
    for t, m in zip(codes, months): assert tmonths[t] == m
    calendar = sorted(set(months)); calendar_index = {m:i for i,m in enumerate(calendar)}
    draws = monthly_draws if label == 'primary' else tournament_draws(len(tids))
    units = np.array([calendar_index[m] for m in tmonths]) if label == 'primary' else np.arange(len(tids))
    n_units = len(calendar) if label == 'primary' else len(tids)
    group = np.array([r['depth_group'] for r in descriptors])
    new = np.array([r['new_players'] for r in descriptors])
    prior_a, prior_b = (np.array([r[key] for r in descriptors]) for key in ['prior_A', 'prior_B'])
    pair = np.array([r['prior_pair'] for r in descriptors])
    arch = (np.array([r['prior_A_archetype'] for r in descriptors]) == 0) | (np.array([r['prior_B_archetype'] for r in descriptors]) == 0)
    profile = {'matches':len(ids), 'tournaments':len(tids), 'months':len(calendar), 'month_start':calendar[0], 'month_end':calendar[-1],
               'one_new_player':int(np.sum(new == 1)), 'both_new_players':int(np.sum(new == 2)),
               'at_least_one_new_player':int(np.sum(new > 0)), 'unseen_pair':int(np.sum(pair == 0)),
               'at_least_one_unseen_player_archetype':int(np.sum(arch)),
               'minimum_prior_count_quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile(np.minimum(prior_a,prior_b),[0,.25,.5,.75,1]).tolist()))}
    specs = [('all','all',np.ones(len(ids),dtype=bool))]
    specs += [('depth',g,group == g) for g in BINS]
    specs += [('new_player',g,new == v) for g,v in [('both_new',2),('one_new',1),('neither_new',0)]]
    results, tournament_rows, composition = [], [], []
    for kind, name, mask in specs:
        ix, counts, means = tournament_means(losses[mask], codes[mask], len(tids))
        result = {'kind':kind,'group':name,'matches':int(mask.sum()),'match_share':float(mask.mean()),
                  'tournaments':len(ix),'months':len(set(months[mask])), 'month_coverage':sorted(set(months[mask])),
                  'scores':{},'comparisons':{}}
        if len(ix):
            for k, model in enumerate(MODELS):
                result['scores'][model] = {'tournament_log_loss':float(means[:,k].mean()),
                    'match_log_loss':float(losses[mask,k].mean()), **probability_metrics(probs[mask,k],y[mask])}
            for comparison, ref, model in CONTRASTS:
                v = paired_interval(means[:,ref]-means[:,model],units[ix],n_units,draws,
                                    reverse_endpoints=label == 'october')
                v['match_weighted_gain'] = float(np.mean(losses[mask,ref]-losses[mask,model]))
                result['comparisons'][comparison] = v
            for j,t in enumerate(ix):
                tournament_rows.append({'period':label,'kind':kind,'group':name,'tournament_id':tids[t], 'month':tmonths[t],
                                        'matches':int(counts[j]),**{f'loss_{k}':float(means[j,k]) for k in range(3)}})
        results.append(result)
        for m in calendar:
            mm = mask & (months == m)
            composition.append({'period':label,'kind':kind,'group':name,'month':m,'matches':int(mm.sum()),
                                'tournaments':len(set(codes[mm])),'month_match_share':float(mm.sum()/np.sum(months == m))})
    assert sum(x['matches'] for x in results if x['kind']=='depth') == len(ids)
    assert sum(x['matches'] for x in results if x['kind']=='new_player') == len(ids)
    both = new == 2
    equality = {MODELS[k]:float(np.max(np.abs(probs[both,k]-probs[both,0]))) if both.any() else None for k in [1,2]}
    # A logit/sigmoid round trip can cross a six-decimal rounding tie.
    # Permit one stored unit plus floating-point subtraction tolerance.
    assert all(v is None or v <= 1e-6 + 1e-12 for v in equality.values())
    return {'profile':profile,'groups':results,'both_new_probability_max_difference':equality,
            'bootstrap_units':calendar if label == 'primary' else tids.tolist()}, tournament_rows, composition, draws


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROC/'history_depth')
    args = parser.parse_args(); out = external(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    began = time.monotonic()
    protocol = PROTOCOL
    assert protocol.is_file(), 'Record the grouping protocol before analysis'
    print('Loading unified history and frozen October population',flush=True)
    history, holdout = load_matches(COLUMNS), load_holdout(COLUMNS)
    expected = {r['match_id']:(r['tournament_id'],r['month'],r['y']) for r in history if r['month']>=START}
    assert len(expected)==sum(r['month']>=START for r in history)==926811
    for r in holdout:
        assert r['match_id'] not in expected
        expected[r['match_id']] = (r['tournament_id'],r['month'],r['y'])
    assert len(holdout)==48288
    descriptors=list(history_descriptors(history,holdout))
    del history,holdout
    write_csv(out/'history_descriptors.csv.gz',descriptors)
    print('Aligning three saved forecasts with descriptors',flush=True)
    probs, losses, prediction_paths = align_predictions(descriptors,expected)
    historical = np.array([r['month']<CUTOFF for r in descriptors])
    draws = month_draws(27)
    summary={'protocol':str(protocol.relative_to(ROOT)), 'protocol_sha256':sha(protocol),
             'models':MODELS,'settings':{'replicates':REPLICATES,'seed':SEED,'block_months':BLOCK,'history_groups':BINS},'periods':{}}
    tournament_rows,composition = [],[]
    for label,mask in [('primary',historical),('october',~historical)]:
        print('Scoring '+label,flush=True)
        desc=[r for r,keep in zip(descriptors,mask) if keep]
        result,trs,comp,multiplicity=summarize_period(label,desc,probs[mask],losses[mask],expected,draws)
        summary['periods'][label]=result;tournament_rows.extend(trs);composition.extend(comp)
        np.save(out/f'{label}_bootstrap_multiplicities.npy',multiplicity,allow_pickle=False)
    ledger_path=PROC/'full_october_2026_run/paper_results.json';ledger=json.loads(ledger_path.read_text())
    for label in ['primary','october']:
        all_result=summary['periods'][label]['groups'][0]
        names=['Card-only','Overall','Player-Skill'] if label=='primary' else ['Card-only','Overall-record control','Player-Skill']
        for model,key in zip(MODELS,names):
            for metric in ['tournament_log_loss','match_log_loss','brier','calibration_rmse']:
                assert abs(all_result['scores'][model][metric]-ledger[label][key][metric])<1e-10,(label,model,metric)
        keys=['Overall vs Card-only','Player-Skill vs Card-only','Player-Skill vs Overall'] if label=='primary' else [None,'Player-Skill vs Card-only','Player-Skill vs Overall-record control']
        for (comparison,_,_),key in zip(CONTRASTS,keys):
            if key is None:continue
            v=all_result['comparisons'][comparison]
            saved=ledger['comparisons']['primary'][key]['gain_ci'] if label=='primary' else ledger['october_comparisons'][key]
            assert np.allclose([v['gain'],*v['ci']],saved,rtol=0,atol=1e-10),(label,comparison)
    summary['validation']={'aligned_historical_matches':int(historical.sum()),'aligned_october_matches':int((~historical).sum()),
                           'all_sample_scores_and_intervals_match_existing_ledger':True,
                           'group_partitions_reconcile':True,'no_model_fitting_or_retuning':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(out/'tournament_scores.csv.gz',tournament_rows)
    write_csv(out/'profile_by_month.csv',composition)
    score_rows, comparison_rows = [], []
    for period,result in summary['periods'].items():
        for group in result['groups']:
            key={'period':period,**{k:group[k] for k in ['kind','group','matches','tournaments','months']}}
            for model,scores in group['scores'].items():
                score_rows.append({**key,'model':model,**{k:v for k,v in scores.items() if k!='bins'}})
            for comparison,values in group['comparisons'].items():
                comparison_rows.append({**key,'comparison':comparison,'gain':values['gain'],
                    'ci_lower':values['ci'][0] if values['ci'] else None,
                    'ci_upper':values['ci'][1] if values['ci'] else None,
                    'match_weighted_gain':values['match_weighted_gain'],
                    'zero_support_replicates':values['zero_support_replicates']})
    write_csv(out/'group_scores.csv',score_rows)
    write_csv(out/'group_comparisons.csv',comparison_rows)
    input_paths=[_resolve(MATCHES),_resolve(HOLDOUT),*prediction_paths,ledger_path]
    code_paths=[Path(__file__),ROOT/'scripts/unified_data.py',ROOT/'scripts/project_paths.py',protocol]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
              'runtime':sys.version,'numpy':importlib.metadata.version('numpy'),'elapsed_seconds':round(time.monotonic()-began,2),
              'input_sha256':{artifact_label(p):sha(p) for p in input_paths},
              'code_and_protocol_sha256':{artifact_label(p):sha(p) for p in code_paths},
              'output_sha256':{name:sha(out/name) for name in [
                  'history_descriptors.csv.gz','summary.json','tournament_scores.csv.gz','profile_by_month.csv',
                  'group_scores.csv','group_comparisons.csv','primary_bootstrap_multiplicities.npy',
                  'october_bootstrap_multiplicities.npy']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for label,result in summary['periods'].items():
        print(label,result['profile'])
        for row in result['groups']:
            if row['kind']=='depth': print(row['group'],row['matches'],row['tournaments'],row['comparisons'])
    print('Saved history-depth analysis and verified full-sample reconciliation',flush=True)


if __name__=='__main__':main()
