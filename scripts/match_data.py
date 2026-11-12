# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Shared integer encodings and archetype-pair counts for current models."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import numpy as np
K_P=50.0
TUNING_MONTHS={"2024-04", "2024-05", "2024-06"}

def prep(rows):
    """Integer-encode players/archetypes once."""
    pl = {p: i for i, p in enumerate(sorted({r["player_A"] for r in rows} | {r["player_B"] for r in rows}))}
    ar = {a: i for i, a in enumerate(sorted({r["archetype_A"] for r in rows} | {r["archetype_B"] for r in rows}))}
    d = {
        "pa": np.array([pl[r["player_A"]] for r in rows]), "pb": np.array([pl[r["player_B"]] for r in rows]),
        "aa": np.array([ar[r["archetype_A"]] for r in rows]), "ab": np.array([ar[r["archetype_B"]] for r in rows]),
        "y": np.array([r["y"] for r in rows], dtype=float),
        "month": np.array([r["month"] for r in rows]),
    }
    d["n_ar"] = len(ar)
    return d


def pair_counts(aa, ab, y, n_ar):
    """Per unordered pair: total decisive matches and wins by the lower-id archetype."""
    lo, hi = np.minimum(aa, ab), np.maximum(aa, ab)
    key = lo * n_ar + hi
    lo_won = np.where(aa == lo, y, 1 - y)
    n = np.bincount(key, minlength=n_ar * n_ar)
    w = np.bincount(key, weights=lo_won, minlength=n_ar * n_ar)
    return n, w


def card_from_counts(aa, ab, n, w, n_ar, loo_y=None):
    """Shrunk pair win rate from A's perspective. loo_y removes the row's own outcome."""
    lo, hi = np.minimum(aa, ab), np.maximum(aa, ab)
    key = lo * n_ar + hi
    nn, ww = n[key].astype(float), w[key].astype(float)
    if loo_y is not None:
        lo_won = np.where(aa == lo, loo_y, 1 - loo_y)
        nn, ww = nn - 1, ww - lo_won
    w_a = np.where(aa == lo, ww, nn - ww)
    p = (w_a + K_P * 0.5) / (nn + K_P)
    return np.where(aa == ab, 0.5, p)


def slicer(d, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}

class Data:
    """Integer-encoded train rows plus (optional) holdout rows sharing the same encoding."""

    def __init__(self, train_rows, hold_rows=()):
        allrows = list(train_rows) + list(hold_rows)
        d = prep(allrows)
        self.n_train = len(train_rows)
        self.n_ar = d["n_ar"]
        self.pa, self.pb, self.aa, self.ab, self.y = d["pa"], d["pb"], d["aa"], d["ab"], d["y"]
        self.month = d["month"].copy()
        self.month[self.n_train:] = "HOLD"
        self.n_pl = int(max(self.pa.max(), self.pb.max())) + 1
        kk = np.concatenate([self.pa.astype(np.int64) * self.n_ar + self.aa,
                             self.pb.astype(np.int64) * self.n_ar + self.ab])
        uk, inv = np.unique(kk, return_inverse=True)
        self.n_keys = len(uk)
        self.kA, self.kB = inv[:len(allrows)], inv[len(allrows):]
        self.tid = np.array([r["tournament_id"] for r in allrows])
        self.mid = np.array([r["match_id"] for r in allrows])
        self.rows = allrows
