# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Incremental October comparisons for manuscript Table 2 and Figure 2.

Reuse saved intervals where available; calculate missing contrasts from the
same saved forecasts and independent-tournament bootstrap. No model is fit.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv
import json
from pathlib import Path

import numpy as np
from scripts.history_depth_analysis import paired_interval, tournament_draws

ROOT = Path(__file__).resolve().parents[1]
PROC = RESULTS


def holdout_comparisons():
    ledger = json.loads((PROC / 'full_october_2026_run/paper_results.json').read_text())
    history = json.loads((PROC / 'history_depth/summary.json').read_text())
    with (PROC / 'FINAL_october_predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r['match_id'] for r in rows}) == 48288
    tids, tc = np.unique([r['tournament_id'] for r in rows], return_inverse=True)
    assert len(tids) == 396
    y = np.array([int(r['observed_y']) for r in rows])
    counts = np.bincount(tc)
    means = {'Constant': np.full(len(tids), np.log(2))}
    for name in ('M1', 'Card-only', 'Overall-record control', 'Player-Skill', 'Joint model'):
        p = np.array([float(r[name]) for r in rows])
        loss = -(y*np.log(np.clip(p, 1e-12, 1)) + (1-y)*np.log(np.clip(1-p, 1e-12, 1)))
        means[name] = np.bincount(tc, weights=loss) / counts
        assert abs(means[name].mean() - ledger['october'][name]['tournament_log_loss']) < 1e-12
    full = next(g for g in history['periods']['october']['groups'] if g['kind'] == 'all')
    overall = full['comparisons']['Overall vs pair']
    saved = {
        'Overall-record control vs Card-only': [overall['gain'], *overall['ci']],
        'Player-Skill vs Overall-record control': ledger['october_comparisons']['Player-Skill vs Overall-record control'],
        'Joint model vs Player-Skill': ledger['october_comparisons']['Joint model vs Player-Skill'],
    }
    draws = tournament_draws(len(tids))
    result = {}
    sequence = [('M1', 'Constant'), ('Card-only', 'M1'), ('Overall-record control', 'Card-only'),
                ('Player-Skill', 'Overall-record control'), ('Joint model', 'Player-Skill')]
    for model, ref in sequence:
        key = model+' vs '+ref
        v = paired_interval(means[ref]-means[model], np.arange(len(tids)), len(tids), draws)
        values = [v['gain'], *v['ci']]
        if key in saved:
            # Preserve the existing ledger's sign-reversal endpoint convention.
            assert abs(values[0]-saved[key][0]) < 1e-12
            assert np.allclose(values[1:], saved[key][1:], atol=1e-5, rtol=0)
            values = saved[key]
        result[key] = values
    return result
