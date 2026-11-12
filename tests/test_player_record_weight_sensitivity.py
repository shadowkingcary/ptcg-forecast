# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Identities and information boundaries for the weight sensitivity experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.archetype_pair import fit_matchup_rate
from scripts.playerskill_general_only import predict_rows
from scripts.player_record_weight_sensitivity import (
    GRID, K_P, forecast_components, historical_components, probabilities, select_weight,
)


def row(mid, month, a='a', b='b', x='x', z='z', y=1):
    return dict(match_id=mid, tournament_id='t'+mid, month=month,
                player_A=a, player_B=b, archetype_A=x, archetype_B=z, y=y)


def test_weight_zero_and_one_match_original_models():
    train = [row('a', '2024-01'), row('b', '2024-02', y=0),
             row('c', '2024-03', b='c'), row('d', '2024-03', x='z', z='z')]
    targets = [row('v', '2024-04'), row('mirror', '2024-04', z='x'),
               row('unknown', '2024-04', a='new', x='newarch')]
    p = probabilities(forecast_components(train, targets, '2024-04'))
    pair = fit_matchup_rate(train, K_P)
    expected0 = [pair(r['archetype_A'], r['archetype_B']) for r in targets]
    assert np.allclose(p[:, 0], expected0, rtol=0, atol=1e-15)
    assert np.allclose(p[:, GRID.index(1.)], predict_rows(train, targets), rtol=0, atol=1e-15)


def test_reversing_sides_complements_all_weight_forecasts():
    train = [row(str(i), '2024-03', y=int(i < 7)) for i in range(10)]
    a = row('v', '2024-04')
    b = row('w', '2024-04', a='b', b='a', x='z', z='x', y=0)
    p = probabilities(forecast_components(train, [a, b], '2024-04'))
    assert np.allclose(p[0] + p[1], 1., rtol=0, atol=1e-15)


def test_unseen_players_have_no_record_adjustment():
    train = [row(str(i), '2024-03') for i in range(8)]
    target = row('v', '2024-04', a='new1', b='new2')
    z = forecast_components(train, [target], '2024-04')
    assert z[0, 1] == 0
    assert np.ptp(probabilities(z)[0]) == 0


def test_validation_uses_fixed_initial_history_and_ignores_later_results():
    history = [row('h', '2024-03'), row('a', '2024-04'),
               row('b', '2024-05'), row('c', '2024-06'), row('later', '2024-07')]
    selection, val, z = select_weight(history)
    assert selection['training_matches'] == 1 and selection['validation_matches'] == 3
    # Identical participants receive identical forecasts throughout validation.
    assert np.array_equal(z, np.repeat(z[:1], 3, axis=0))
    changed = [{**r, 'y': 1-r['y']} if r['month'] >= '2024-07' else r for r in history]
    changed += [row('oct', '2026-10', y=0)]
    other, _, oz = select_weight(changed)
    assert selection == other and np.array_equal(z, oz)
    # An exact tie uses the original coefficient of one.
    tied = [history[0]] + [row(m, m, a='unseen1', b='unseen2')
                           for m in ('2024-04', '2024-05', '2024-06')]
    assert select_weight(tied)[0]['weight'] == 1.


def test_monthly_history_excludes_entire_current_month():
    history = [row('warm', '2024-06'), row('j1', '2024-07'),
               row('j2', '2024-07'), row('aug', '2024-08')]
    rows, z, audit = historical_components(history)
    assert np.array_equal(z[0], z[1])
    assert [a['training_matches'] for a in audit] == [1, 3]
    changed = [r if r['month'] != '2024-07' else {**r, 'y': 1-r['y']} for r in history]
    _, cz, _ = historical_components(changed)
    assert np.array_equal(z[:2], cz[:2])
    assert not np.array_equal(z[2], cz[2])
    try:
        forecast_components(history, [history[1]], '2024-07')
    except ValueError:
        pass
    else:
        raise AssertionError('Current-month training outcomes accepted')


def test_october_targets_need_no_outcomes_and_never_update_history():
    history = [row('warm', '2026-09')]
    hold = [row('o1', '2026-10'), row('o2', '2026-10', y=0)]
    stripped = [{k: v for k, v in r.items() if k != 'y'} for r in hold]
    z = forecast_components(history, hold, '2026-10')
    assert np.array_equal(z, forecast_components(history, stripped, '2026-10'))
    assert np.array_equal(z[0], z[1])
    for function in (lambda: historical_components(history+hold),
                     lambda: forecast_components(history+hold[:1], hold[1:], '2026-10')):
        try:
            function()
        except ValueError:
            pass
        else:
            raise AssertionError('October outcomes entered training')
