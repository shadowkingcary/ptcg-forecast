# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""
Estimate archetype strength,
archetype-pair, player, and player-archetype effects JOINTLY in one ridge
logistic regression (partial pooling via per-block prior scales), instead of
adding independently estimated player adjustments to a separately estimated
archetype probability.

  logit p(A/X vs B/Y) = (theta_X - theta_Y)            archetype strength  (prior sd s_theta = 1)
                      + sgn * delta_{XY}                pair-specific       (prior sd s_pair)
                      + (alpha_A - alpha_B)             player              (prior sd s_player)
                      + (gamma_{A,X} - gamma_{B,Y})     player-archetype    (prior sd s_pa)

Antisymmetric design, no intercept, so p(reverse) = 1 - p. A feature column of
scale s with unit ridge penalty is equivalent to a prior sd of s on its
coefficient. Unseen effects contribute zero. Prior scales are chosen by
coordinate search around (0.28, 0.45, 0.28) on the Apr-Jun 2024 validation
months (training through Mar 2024) and then frozen.

Output: the external results directory, FINAL_joint_predictions.csv, FINAL_joint_targeted_bootstrap.csv
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, statistics, sys, warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.logistic_baseline import MIN_TRAIN_MONTH_EXCLUSIVE  # noqa: E402
from scripts.score_utils import block_bootstrap, load_ll, tournament_stats  # noqa: E402
from scripts.match_data import prep, slicer, TUNING_MONTHS  # noqa: E402
from scripts.unified_data import load_matches  # noqa: E402
COLUMNS = ["tournament_id", "match_id", "player_A", "player_B", "archetype_A", "archetype_B", "month", "y"]  # the unified-table columns this script reads

warnings.filterwarnings("ignore")
PROC = RESULTS
S_THETA = 1.0
BASE = (0.28, 0.45, 0.28)  # (pair, player, player-archetype)
MAX_ITER = 10000  # iteration cap; the fit must stop on the tolerance below, never on this cap
TOL = 1e-8  # tight L-BFGS tolerance: the fit is then the unique ridge optimum, independent of column order


class _Idx:
    """Sorted-unique lookup: get_indexer returns position or -1."""
    def __init__(self, a): self.a = a
    def get_indexer(self, q):
        q = np.asarray(q); pos = np.searchsorted(self.a, q); pos = np.minimum(pos, len(self.a) - 1)
        return np.where(self.a[pos] == q, pos, -1)


def design(d, maps, n_ar, scales):
    """Sparse antisymmetric design using column maps built from the training set."""
    s_pair, s_pl, s_pa = scales
    n = len(d["y"]); rows, cols, vals = [], [], []
    off = {"theta": 0, "pair": n_ar, "pl": n_ar + maps["n_pair"], "pa": n_ar + maps["n_pair"] + maps["n_pl"]}
    idx = np.arange(n)
    def add(r, c, v, ok=None):
        if ok is not None: r, c, v = r[ok], c[ok], v[ok]
        rows.append(r); cols.append(c); vals.append(v)
    add(idx, d["aa"] + off["theta"], np.full(n, S_THETA)); add(idx, d["ab"] + off["theta"], np.full(n, -S_THETA))
    lo, hi = np.minimum(d["aa"], d["ab"]), np.maximum(d["aa"], d["ab"])
    pk = lo * n_ar + hi
    pc = maps["pair"].get_indexer(pk); sgn = np.where(d["aa"] < d["ab"], 1.0, -1.0)
    add(idx, pc + off["pair"], sgn * s_pair, ok=(pc >= 0) & (d["aa"] != d["ab"]))
    for side, sgnp in (("pa", 1.0), ("pb", -1.0)):
        c = maps["pl"].get_indexer(d[side]); add(idx, c + off["pl"], np.full(n, sgnp * s_pl), ok=c >= 0)
    for (pside, aside), sgnp in ((("pa", "aa"), 1.0), (("pb", "ab"), -1.0)):
        c = maps["pa"].get_indexer(d[pside].astype(np.int64) * n_ar + d[aside])
        add(idx, c + off["pa"], np.full(n, sgnp * s_pa), ok=c >= 0)
    R, C, V = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    return sparse.csr_matrix((V, (R, C)), shape=(n, off["pa"] + maps["n_pa"]))


def build_maps(tr, n_ar):
    lo, hi = np.minimum(tr["aa"], tr["ab"]), np.maximum(tr["aa"], tr["ab"])
    pairs = np.unique((lo * n_ar + hi)[tr["aa"] != tr["ab"]])
    pls = np.unique(np.concatenate([tr["pa"], tr["pb"]]))
    pas = np.unique(np.concatenate([tr["pa"].astype(np.int64) * n_ar + tr["aa"], tr["pb"].astype(np.int64) * n_ar + tr["ab"]]))
    return {"pair": _Idx(pairs), "pl": _Idx(pls), "pa": _Idx(pas),
            "n_pair": len(pairs), "n_pl": len(pls), "n_pa": len(pas)}


def fit_predict(tr, te, n_ar, scales):
    maps = build_maps(tr, n_ar)
    X = design(tr, maps, n_ar, scales)
    m = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs", max_iter=MAX_ITER, tol=TOL)
    m.fit(X, tr["y"])
    assert int(m.n_iter_[0]) < MAX_ITER, "joint-model fit hit the iteration cap without converging"
    return m.predict_proba(design(te, maps, n_ar, scales))[:, 1]


def tune(d):
    n_ar = d["n_ar"]; tr = slicer(d, d["month"] < MIN_TRAIN_MONTH_EXCLUSIVE)
    te = slicer(d, np.isin(d["month"], list(TUNING_MONTHS)))
    configs = [BASE] + [tuple(b * (f if i == j else 1.0) for i, b in enumerate(BASE)) for j in range(3) for f in (0.5, 2.0)]
    best, bl = None, 1e9
    for sc in configs:
        p = fit_predict(tr, te, n_ar, sc); y = te["y"]
        ml = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        print(f"  scales(pair,player,pa)={tuple(round(x,3) for x in sc)}: validation mean log loss {ml:.5f}", flush=True)
        if ml < bl: bl, best = ml, sc
    print("Selected", best); return best


def main():
    rows = load_matches(COLUMNS); d = prep(rows); n_ar = d["n_ar"]
    skip = "--skip-fit" in sys.argv  # reuse saved predictions; recompute comparisons only
    if skip:
        sc = (0.28, 0.45, 0.14)  # selected on the validation months in the original run
        out = list(csv.DictReader(open(PROC / "FINAL_joint_predictions.csv")))
    else:
        print("=== tuning prior scales ===")
        sc = tune(d)
        out = []
    for m in ([] if skip else [x for x in sorted(set(d["month"])) if x >= MIN_TRAIN_MONTH_EXCLUSIVE]):
        tr = slicer(d, d["month"] < m); mask = d["month"] == m; te = slicer(d, mask)
        p = fit_predict(tr, te, n_ar, sc)
        if m not in TUNING_MONTHS:
            for i, pi in zip(np.where(mask)[0], p):
                r = rows[i]; yi = int(r["y"]); pi = min(max(float(pi), 1e-12), 1 - 1e-12)
                out.append({"tournament_id": r["tournament_id"], "month": m, "match_id": r["match_id"], "observed_y": yi,
                            "prob": round(pi, 6), "logloss": round(-(yi * math.log(pi) + (1 - yi) * math.log(1 - pi)), 6)})
        print(f"{m}: train_n={len(tr['y'])} done", flush=True)
    if not skip:
        with (PROC / "FINAL_joint_predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    conv = [{"tournament_id": r["tournament_id"], "month": r["month"], "match_id": r["match_id"], "hps_logloss": float(r["logloss"])} for r in out]
    refs = {"Player-Skill pooled": load_ll(PROC / "FINAL_playerskill_pooled_predictions.csv", "pooled_logloss"),
            "Overall records": load_ll(PROC / "FINAL_playerskill_generalonly_predictions.csv", "logloss"),
            "M1": load_ll(PROC / "FINAL_hier_full_predictions.csv", "m1_logloss")}
    with (PROC / "FINAL_joint_targeted_bootstrap.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "n_tournaments", "joint_log_loss", "reference_log_loss", "improvement", "ci_lo", "ci_hi", "scales"])
        for name, ref in refs.items():
            st = tournament_stats(conv, ref); obs, lo, hi = block_bootstrap(st, lambda s: s[1] - s[2])
            rm, jm = statistics.mean(s[1] for s in st), statistics.mean(s[2] for s in st)
            print(f"joint vs {name}: {rm:.6f} -> {jm:.6f}; improvement {obs:.6f} [{lo:.6f}, {hi:.6f}]")
            w.writerow([name, len(st), round(jm, 6), round(rm, 6), round(obs, 6), round(lo, 6), round(hi, 6), str(tuple(round(x, 3) for x in sc))])

if __name__ == "__main__":
    main()
