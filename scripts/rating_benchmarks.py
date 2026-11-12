# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Elo plus archetype-pair benchmark. Select K and coefficient on April–June
2024 tournament-weighted loss. Read month-start ratings, update after historical
tournaments, and freeze every October forecast before October 1."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv
import math
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.match_data import pair_counts, card_from_counts  # noqa: E402
from scripts.match_data import Data  # noqa: E402
from scripts.score_utils import block_bootstrap, load_ll, tournament_stats  # noqa: E402
from scripts.archetype_pair import fit_matchup_rate  # noqa: E402
from scripts.player_records import fit_player_skill_pooled, logit as logit_s, sigmoid as sigmoid_s  # noqa: E402
from scripts.unified_data import load_matches, load_holdout  # noqa: E402
COLUMNS = ["tournament_id", "tournament_date", "match_id", "player_A", "player_B", "archetype_A", "archetype_B", "winner", "month", "y"]  # the unified-table columns this script reads

PROC = RESULTS
LN10 = math.log(10.0)
K_GRID = [8, 16, 24, 32, 48, 64]
C_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
VAL_MONTHS = ["2024-04", "2024-05", "2024-06"]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ setup
def setup():
    train = load_matches(COLUMNS)
    hold = load_holdout(COLUMNS)
    for r in hold:
        r["y"] = 1 if r["winner"] == "A" else 0; r["month"] = "2026-10"
    D = Data(train, hold)
    N = len(D.y)
    dates = [r["tournament_date"] for r in D.rows]
    groups = defaultdict(list)
    for i, t in enumerate(D.tid):
        groups[t].append(i)
    order = sorted(groups, key=lambda t: (dates[groups[t][0]], t))
    groups = {t: np.array(v) for t, v in groups.items()}
    tmonth = {t: dates[groups[t][0]][:7] for t in order}
    tday = {t: float((np.datetime64(dates[groups[t][0]][:19]) - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "D")) for t in order}
    mstart = lambda m: float((np.datetime64(m + "-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    # archetype baseline (counts strictly before the month), identical to Player-Skill's card component
    n_ar = D.n_ar; npair = np.zeros(n_ar * n_ar); wpair = np.zeros(n_ar * n_ar); base = np.zeros(N)
    for m in sorted(set(D.month[:D.n_train])):
        idx = np.where(D.month == m)[0]
        c = card_from_counts(D.aa[idx], D.ab[idx], npair, wpair, n_ar); base[idx] = np.log(c / (1 - c))
        n_m, w_m = pair_counts(D.aa[idx], D.ab[idx], D.y[idx], n_ar); npair += n_m; wpair += w_m
    # October is absent from training. Freeze its archetype component after September.
    hi = np.arange(D.n_train, N)
    c2 = card_from_counts(D.aa[hi], D.ab[hi], npair, wpair, n_ar)
    base[hi] = np.log(c2 / (1 - c2))
    _, tcode = np.unique(D.tid, return_inverse=True)
    month_eff = D.month.copy(); month_eff[D.n_train:] = "2026-10"
    return D, order, groups, tmonth, tday, mstart, base, tcode, month_eff


# ------------------------------------------------------------------ rating passes
def elo_pass(D, order, groups, tmonth, K):
    N = len(D.y); R = np.full(D.n_pl, 1500.0); Rs = R.copy(); cur = None
    zs = np.zeros(N); zf = np.zeros(N)
    for t in order:
        ix = groups[t]; m = tmonth[t]
        if m != cur:
            Rs = R.copy(); cur = m
        pa, pb, y = D.pa[ix], D.pb[ix], D.y[ix]
        dsf = R[pa] - R[pb]
        zf[ix] = dsf * LN10 / 400.0; zs[ix] = (Rs[pa] - Rs[pb]) * LN10 / 400.0
        if np.any(ix >= D.n_train):
            assert np.all(ix >= D.n_train), "Tournament crosses the holdout boundary"
            zf[ix] = zs[ix]  # October uses the frozen month-start state in both readings.
            continue
        E = 1.0 / (1.0 + 10.0 ** (-dsf / 400.0)); d = K * (y - E)
        R += np.bincount(pa, weights=d, minlength=D.n_pl) - np.bincount(pb, weights=d, minlength=D.n_pl)
    return zs, zf




# ------------------------------------------------------------------ evaluation helpers
def tourn_loss(D, tcode, idx, p):
    y = D.y[idx]; p = np.clip(p, 1e-12, 1 - 1e-12); ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    tc = tcode[idx]; u, inv = np.unique(tc, return_inverse=True)
    return float((np.bincount(inv, weights=ll) / np.bincount(inv)).mean())


def predict(variant, z, base, c):
    if variant == "conv":
        return sig(z)
    if variant == "scaled":
        return sig(c * z)
    return sig(base + c * z)  # "arch"


def select(D, tcode, month_eff, Z, base, months, label):
    idx = np.where(np.isin(month_eff[:D.n_train], months))[0]
    out = {}
    for fam, cfgs in (("Elo", K_GRID),):
        for variant in ("arch",):
            best, bl = None, 1e9
            for cfg in cfgs:
                for c in ([1.0] if variant == "conv" else C_GRID):
                    p = predict(variant, Z[(fam, cfg)][0][idx], base[idx], c)
                    l = tourn_loss(D, tcode, idx, p)
                    if l < bl: bl, best = l, (cfg, c)
            out[(fam, variant)] = best
            print(f"[{label}] selected {fam} {variant}: param={best[0]} c={best[1]}  (validation tournament log loss {bl:.6f})", flush=True)
    return out


def rmse(p, y):
    b = np.minimum((p * 10).astype(int), 9)
    return math.sqrt(sum((b == k).sum() * (p[b == k].mean() - y[b == k].mean()) ** 2 for k in np.unique(b)) / len(p))


def main():
    D, order, groups, tmonth, tday, mstart, base, tcode, month_eff = setup()
    N = len(D.y); print(f"{D.n_train} train matches + {N - D.n_train} holdout; {len(order)} tournaments", flush=True)
    Z = {}
    for K in K_GRID:
        Z[("Elo", K)] = elo_pass(D, order, groups, tmonth, K)

    sel_primary = select(D, tcode, month_eff, Z, base, VAL_MONTHS, "primary: Apr-Jun 2024")
    with open(PROC / "FINAL_rating_benchmarks_selection.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["selection_window", "family", "variant", "K_or_tau", "c"])
        for lab, sel in (("2024-04..2024-06", sel_primary),):
            for (fam, var), (cfg, c) in sel.items(): w.writerow([lab, fam, var, cfg, c])

    # ---------------- evaluation (parameters now frozen) ----------------
    train_idx = np.arange(D.n_train); hold_idx = np.arange(D.n_train, N)
    prim = train_idx[(month_eff[:D.n_train] >= "2024-07")]
    later = train_idx[(month_eff[:D.n_train] >= "2025-01")]
    refs = {"Player-Skill": load_ll(PROC / "FINAL_playerskill_pooled_predictions.csv", "pooled_logloss"),
            "Player-Skill, overall record only": load_ll(PROC / "FINAL_playerskill_generalonly_predictions.csv", "logloss"),
            "card-only": load_ll(PROC / "FINAL_cardonly_vs_playerskill_predictions.csv", "cardonly_logloss")}
    NAMES = {"conv": "", "scaled": ", scaled", "arch": " + archetype"}
    res = []
    pred_store = {}

    def evaluate(specname, sel, idx, reading, window):
        for (fam, var), (cfg, c) in sel.items():
            z = Z[(fam, cfg)][0 if reading == "snapshot" else 1]
            p = predict(var, z[idx], base[idx], c); model = f"{fam}{NAMES[var]}"
            if specname == "primary" and reading == "snapshot":
                pred_store[model] = (idx, p)
            ll = -(D.y[idx] * np.log(np.clip(p, 1e-12, 1)) + (1 - D.y[idx]) * np.log(np.clip(1 - p, 1e-12, 1)))
            conv = [{"tournament_id": D.tid[i], "month": month_eff[i], "match_id": D.mid[i], "hps_logloss": float(l)} for i, l in zip(idx, ll)]
            for refname, ref in refs.items():
                st = tournament_stats(conv, ref)
                obs, lo, hi = block_bootstrap(st, lambda s: s[1] - s[2])  # ref - model
                res.append([specname, reading, window, model, cfg, c, refname, len(st),
                            round(statistics.mean(s[2] for s in st), 6), round(statistics.mean(s[1] for s in st), 6),
                            round(-obs, 6), round(-hi, 6), round(-lo, 6), round(rmse(p, D.y[idx]), 6)])

    evaluate("primary", sel_primary, prim, "snapshot", "2024-07..2026-09")

    # ---------------- holdout (frozen, trained on months < 2026-10; naive bootstrap) ----------------
    train_pre = [r for r in D.rows[:D.n_train] if r["month"] < "2026-10"]
    skill = fit_player_skill_pooled(train_pre, 20, 50); card = fit_matchup_rate(train_pre, 50)
    hold_rows = D.rows[D.n_train:]
    ps = np.array([sigmoid_s(logit_s(card(r["archetype_A"], r["archetype_B"])) + skill(r["player_A"], r["archetype_A"]) - skill(r["player_B"], r["archetype_B"])) for r in hold_rows])
    gen = {r["match_id"]: float(r["prob"]) for r in csv.DictReader(open(PROC / "FINAL_playerskill_generalonly_holdout_predictions.csv"))}
    psg = np.array([gen[m] for m in D.mid[hold_idx]])
    yh = D.y[hold_idx]; th = D.tid[hold_idx]; ut = np.unique(th)
    def hl(p):
        p = np.clip(p, 1e-12, 1 - 1e-12); l = -(yh * np.log(p) + (1 - yh) * np.log(1 - p)); return np.array([l[th == t].mean() for t in ut])
    refh = {"Player-Skill": hl(ps), "Player-Skill, overall record only": hl(psg)}
    rng = np.random.default_rng(20261011); ix = rng.integers(0, len(ut), (5000, len(ut)))
    hold_out = []
    for reading in ("snapshot",):  # October is strictly held out for every model.
        for (fam, var), (cfg, c) in sel_primary.items():
            z = Z[(fam, cfg)][0 if reading == "snapshot" else 1]; p = predict(var, z[hold_idx], base[hold_idx], c); lm = hl(p)
            for refname, rl in refh.items():
                d = lm - rl; bs = np.sort(d[ix].mean(1))
                hold_out.append([reading, f"{fam}{NAMES[var]}", refname, round(float(lm.mean()), 6), round(float(rl.mean()), 6), round(float(d.mean()), 6), round(float(bs[125]), 6), round(float(bs[4875]), 6), round(rmse(p, yh), 6)])

    with open(PROC / "FINAL_rating_benchmarks_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spec", "reading", "window", "model", "K_or_tau", "c", "reference", "n_tournaments", "model_log_loss", "reference_log_loss", "model_minus_reference", "ci_lo", "ci_hi", "calibration_rmse"]); w.writerows(res)
    with open(PROC / "FINAL_rating_benchmarks_holdout.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reading", "model", "reference", "model_log_loss", "reference_log_loss", "model_minus_reference", "ci_lo_naive", "ci_hi_naive", "calibration_rmse"]); w.writerows(hold_out)

    # ---------------- report ----------------
    print("\n=== RESULTS (model minus reference; negative favours the rating system) ===")
    for r in res:
        if r[0] == "primary" and r[1] == "snapshot" and r[6] == "Player-Skill":
            print(f"PRIMARY snapshot  {r[3]:24s} loss {r[8]:.6f}  vs Player-Skill {r[10]:+.6f} [{r[11]:+.6f}, {r[12]:+.6f}]  RMSE {r[13]:.4f}  (param {r[4]}, c {r[5]})")
    for r in res:
        if r[0] == "primary" and r[1] == "snapshot" and r[6] == "Player-Skill, overall record only":
            print(f"PRIMARY snapshot  {r[3]:24s} vs PS overall-only {r[10]:+.6f} [{r[11]:+.6f}, {r[12]:+.6f}]")
    print("\n=== HOLDOUT (snapshot) ===")
    for h in hold_out:
        if h[0] == "snapshot" and h[2] == "Player-Skill":
            print(f"{h[1]:24s} loss {h[3]:.6f} (PS {h[4]:.6f}) diff {h[5]:+.6f} [{h[6]:+.6f}, {h[7]:+.6f}] RMSE {h[8]:.4f}")


if __name__ == "__main__":
    main()
