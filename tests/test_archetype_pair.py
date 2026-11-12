# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Archetype-pair symmetry and smoothing checks."""
import sys
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.archetype_pair import fit_matchup_rate

def _row(a, b, y):
    return {"archetype_A": a, "archetype_B": b, "y": y}

def _sample_train_rows():
    return (
        [_row("x", "y", 1)] * 7 + [_row("x", "y", 0)] * 3
        + [_row("y", "x", 1)] * 2 + [_row("y", "x", 0)] * 1
        + [_row("m", "m", 1)] * 5 + [_row("m", "m", 0)] * 5
    )

def test_pairwise_zero_history_is_half():
    predict = fit_matchup_rate(_sample_train_rows(), k=10)
    assert predict("unseen1", "unseen2") == 0.5

def test_pairwise_mirror_is_always_half():
    predict = fit_matchup_rate(_sample_train_rows(), k=10)
    assert predict("m", "m") == 0.5

def test_pairwise_mirror_half_even_with_lopsided_history():
    train = [_row("m", "m", 1)] * 19 + [_row("m", "m", 0)] * 1
    predict = fit_matchup_rate(train, k=10)
    assert predict("m", "m") == 0.5

def test_pairwise_reversed_pair_is_complementary():
    predict = fit_matchup_rate(_sample_train_rows(), k=10)
    p_xy = predict("x", "y")
    p_yx = predict("y", "x")
    assert abs((p_xy + p_yx) - 1.0) < 1e-9

def test_pairwise_pools_both_orientations():
    # x-vs-y (x wins 7, y wins 3) and y-vs-x (y wins 2, x wins 1) must pool into
    # one pair count: x wins 7+1=8, y wins 3+2=5, n=13.
    predict = fit_matchup_rate(_sample_train_rows(), k=0.0001)
    p_xy = predict("x", "y")
    assert abs(p_xy - 8 / 13) < 1e-3
