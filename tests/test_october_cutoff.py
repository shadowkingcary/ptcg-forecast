# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Regression checks for a full held-out month absent from the training table."""
import csv
import sys
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_data import SCHEMA, load_matches, load_holdout
from scripts.rating_benchmarks import setup, elo_pass
from scripts.rating_fresh_comparison import ps_fresh
from scripts.archetype_pair import fit_matchup_rate


def example(day, winner="A"):
    return dict(tournament_id="t" + day, tournament_date=day + "T12:00:00.000Z", era="4",
                match_id="m" + day, player_A="pA", player_B="pB", archetype_A="x", archetype_B="y",
                winner=winner, round="1", month=day[:7], y=int(winner == "A"))


def test_loader_excludes_all_october_and_rejects_outside_holdout_dates():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "matches.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, SCHEMA, extrasaction="ignore"); w.writeheader()
            w.writerows([example("2026-09-30"), example("2026-10-01"), example("2026-11-01")])
        assert load_matches(["match_id"], path=p) == [{"match_id": "m2026-09-30"}]
        try:
            load_holdout(["match_id"], path=p)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-period holdout rows accepted")




def test_rating_hybrid_uses_september_counts_for_october():
    train = [example("2026-09-01"), example("2026-09-02")]
    hold = [example("2026-10-01", "B"), example("2026-10-31", "B")]
    with patch("scripts.rating_benchmarks.load_matches", return_value=train), patch("scripts.rating_benchmarks.load_holdout", return_value=hold):
        d, _, _, _, _, _, base, _, _ = setup()
    expected = fit_matchup_rate(train, 50)("x", "y")
    assert np.allclose(1 / (1 + np.exp(-base[d.n_train:])), expected, atol=1e-14)
    assert expected > .5


def test_both_rating_readings_ignore_every_october_outcome():
    train = [example("2026-09-01"), example("2026-09-02")]
    hold = [example("2026-10-01"), example("2026-10-10"), example("2026-10-31")]
    with patch("scripts.rating_benchmarks.load_matches", return_value=train), patch("scripts.rating_benchmarks.load_holdout", return_value=hold):
        d, order, groups, months, days, mstart, _, _, _ = setup()
    original_y = d.y.copy()
    for run_rating in (lambda: elo_pass(d, order, groups, months, 16),):
        d.y[:] = original_y
        a = run_rating()
        d.y[d.n_train:] = 1 - original_y[d.n_train:]
        b = run_rating()
        assert all(np.array_equal(p, q) for p, q in zip(a, b))
        assert np.array_equal(a[0][d.n_train:], a[1][d.n_train:])
        assert a[1][1] > 0  # Historical tournament updating still learns from September.


def test_tournament_player_counts_remain_frozen_through_october():
    train = [example("2026-09-01"), example("2026-09-02")]
    hold = [example("2026-10-01"), example("2026-10-10"), example("2026-10-31")]
    with patch("scripts.rating_benchmarks.load_matches", return_value=train), patch("scripts.rating_benchmarks.load_holdout", return_value=hold):
        d, order, groups, _, _, _, _, _, _ = setup()
    p = ps_fresh(d, order, groups)
    d.y[d.n_train:] = 1 - d.y[d.n_train:]
    q = ps_fresh(d, order, groups)
    assert np.array_equal(p, q)
    assert np.all(p[d.n_train:] == p[d.n_train])
    assert p[1] > p[0]  # September results still update the historical experiment.
