# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Check weighting, temporal boundaries, and safe forecast reuse."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT))
from scripts import tuning_objective_sensitivity as experiment


def row(mid, month, y=1, tid=None):
    return dict(match_id=mid, tournament_id=tid or mid, month=month,
                player_A='a', player_B='b', archetype_A='x', archetype_B='z', y=y)


def test_weighting_objectives_can_select_different_candidates():
    # Three matches in a large event and one match in a small event.
    tids = ['large'] * 3 + ['small']
    a = {'name': 'a', **experiment.objective_scores(np.array([0., 0., 0., 1.]), tids)}
    b = {'name': 'b', **experiment.objective_scores(np.full(4, .4), tids)}
    assert a['match_log_loss'] == .25 and a['tournament_log_loss'] == .5
    assert experiment.choose([a, b], 'match')['name'] == 'a'
    assert experiment.choose([a, b], 'tournament')['name'] == 'b'
    assert experiment.choose([b, {**b, 'name': 'tied'}], 'match')['name'] == 'b'


def test_validation_predictions_use_fixed_initial_history_and_ignore_later_outcomes():
    history = [row('train', '2024-03'), row('apr', '2024-04'),
               row('may', '2024-05'), row('jun', '2024-06'), row('later', '2024-07')]
    calls = []
    def stub(train, targets, model, settings, cutoff):
        calls.append((len(train), [r['month'] for r in targets], cutoff))
        assert [r['match_id'] for r in train] == ['train']
        return np.full(len(targets), .6)
    with patch.object(experiment, 'forecast', stub), redirect_stdout(StringIO()):
        first, _, _ = experiment.select_settings(history)
        changed = [({**r, 'y': 0} if r['month'] >= '2024-07' else r) for r in history]
        second, _, _ = experiment.select_settings(changed + [row('oct', '2026-10')])
    assert first == second
    assert len(calls) == 58
    assert all(n == 1 and months == list(experiment.VALIDATION) and cutoff == '2024-04'
               for n, months, cutoff in calls)


def test_pair_choice_propagates_to_player_smoothing_grid():
    history = [row('train', '2024-03')]
    val = [row('a', '2024-04', tid='large'), row('b', '2024-04', tid='large'),
           row('c', '2024-04', tid='large'), row('d', '2024-05'), row('e', '2024-06')]
    calls = []
    def stub(train, targets, model, settings, cutoff):
        calls.append((model, settings.copy()))
        if model == 'pair':
            return (np.array([.99]*3 + [.45]*2) if settings['k_p'] == 2
                    else np.full(5, .7 if settings['k_p'] == 5 else .1))
        return np.full(len(targets), .6)
    with patch.object(experiment, 'forecast', stub), redirect_stdout(StringIO()):
        selected, _, _ = experiment.select_settings(history + val)
    assert selected['settings']['match']['pair']['k_p'] == 2
    assert selected['settings']['tournament']['pair']['k_p'] == 5
    assert {s['k_p'] for model, s in calls if model == 'archetype_specific'} == {2, 5}
    for objective in experiment.OBJECTIVES:
        settings = selected['settings'][objective]
        assert settings['overall']['k_p'] == settings['pair']['k_p']
        assert settings['overall']['k_o'] == settings['archetype_specific']['k_o']


def test_forecasts_do_not_read_target_outcomes_or_use_october_training():
    train = [row('a', '2026-09'), row('b', '2026-09', y=0)]
    hold = [row('first', '2026-10'), row('last', '2026-10', y=0)]
    stripped = [{k: v for k, v in r.items() if k != 'y'} for r in hold]
    settings = {'k_p': 50, 'k_o': 20, 'k_s': 50}
    for model in ('pair', 'overall', 'archetype_specific'):
        a = experiment.forecast(train, hold, model, settings, '2026-10')
        b = experiment.forecast(train, stripped, model, settings, '2026-10')
        assert np.array_equal(a, b) and a[0] == a[1]
    def joint_stub(tr, te, n_ar, scales):
        assert np.array_equal(tr['y'], [1, 0])
        assert np.all(te['y'] == 0)
        return np.full(len(te['y']), .5)
    with patch.object(experiment, 'joint_predict', joint_stub):
        assert np.array_equal(experiment.forecast(train, stripped, 'joint',
                              {'scales': [.28, .45, .14]}, '2026-10'), [.5, .5])
    try:
        experiment.forecast(train + hold[:1], hold[1:], 'pair', settings, '2026-10')
    except ValueError:
        pass
    else:
        raise AssertionError('October outcomes entered fitting')


def test_changed_settings_cannot_reuse_original_forecasts():
    expected = {m: {'value': 1} for m in experiment.MODELS}
    selection = {'settings': {'match': deepcopy(expected), 'tournament': deepcopy(expected)},
                 'changed_models': []}
    experiment.require_reusable_forecasts(selection, expected)
    for objective in experiment.OBJECTIVES:
        bad = deepcopy(selection)
        bad['settings'][objective]['pair']['value'] = 2
        try:
            experiment.require_reusable_forecasts(bad, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Different selected settings reused baseline scores')
