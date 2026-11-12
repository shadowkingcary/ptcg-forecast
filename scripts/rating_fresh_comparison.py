# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Archetype-specific records with tournament updates, compared with monthly
updates on the historical window. Fixed k_p=50, k_o=20, k_s=50. October stays frozen."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.rating_benchmarks import setup, sig, PROC
from scripts.match_data import pair_counts, card_from_counts
from scripts.score_utils import block_bootstrap, load_ll, tournament_stats


def ps_fresh(D, order, groups):
    n_ar = D.n_ar; N = len(D.y)
    npair = np.zeros(n_ar * n_ar); wpair = np.zeros(n_ar * n_ar)
    ot = np.zeros(D.n_pl); ow = np.zeros(D.n_pl); at = np.zeros(D.n_keys); aw = np.zeros(D.n_keys)
    p = np.zeros(N)
    for t in order:
        ix = groups[t]; pa, pb, y = D.pa[ix], D.pb[ix], D.y[ix]; kA, kB = D.kA[ix], D.kB[ix]
        card = card_from_counts(D.aa[ix], D.ab[ix], npair, wpair, n_ar)
        def skill(pl, k):
            overall = (ow[pl] + 10.0) / (ot[pl] + 20.0)
            q = np.clip((aw[k] + 50.0 * overall) / (at[k] + 50.0), 1e-9, 1 - 1e-9)
            return np.log(q / (1 - q))
        p[ix] = sig(np.log(card / (1 - card)) + skill(pa, kA) - skill(pb, kB))
        if np.any(ix >= D.n_train):
            assert np.all(ix >= D.n_train), "Tournament crosses the holdout boundary"
            continue  # All October outcomes remain excluded from both count components.
        np.add.at(ot, pa, 1); np.add.at(ot, pb, 1); np.add.at(ow, pa, y); np.add.at(ow, pb, 1 - y)
        np.add.at(at, kA, 1); np.add.at(at, kB, 1); np.add.at(aw, kA, y); np.add.at(aw, kB, 1 - y)
        n_m, w_m = pair_counts(D.aa[ix], D.ab[ix], y, n_ar); npair += n_m; wpair += w_m
    return p


def main():
    D, order, groups, tmonth, tday, mstart, base, tcode, month_eff = setup()
    pf = ps_fresh(D, order, groups)
    out = []
    train_idx = np.arange(D.n_train)
    def stats_vs(idx, p, ref):
        y = D.y[idx]; ll = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
        return ll
    def compare(name, idx, p_model, p_ref, ref_name, window, spec):
        lm = stats_vs(idx, p_model, None); lr = stats_vs(idx, p_ref, None)
        conv = [{"tournament_id": D.tid[i], "month": month_eff[i], "match_id": D.mid[i], "hps_logloss": float(l)} for i, l in zip(idx, lm)]
        ref = {D.mid[i]: float(l) for i, l in zip(idx, lr)}
        st = tournament_stats(conv, ref); obs, lo, hi = block_bootstrap(st, lambda s: s[1] - s[2])
        row = [spec, window, name, ref_name, len(st), round(sum(s[2] for s in st) / len(st), 6), round(sum(s[1] for s in st) / len(st), 6), round(-obs, 6), round(-hi, 6), round(-lo, 6)]
        out.append(row); print(row, flush=True)
    prim = train_idx[month_eff[:D.n_train] >= "2024-07"]
    # PS monthly (existing) vs PS fresh: the value of freshness itself
    rows = {r["match_id"]: float(r["pooled_prob"]) for r in csv.DictReader(open(PROC / "FINAL_playerskill_pooled_predictions.csv"))}
    pm = np.array([rows[D.mid[i]] for i in prim])
    compare("Player-Skill (tournament-fresh)", prim, pf[prim], pm, "Player-Skill (monthly)", "2024-07..2026-09", "primary")
    # Update-frequency experiments end in September; October is strictly held out.
    with open(PROC / "FINAL_rating_fresh_comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spec", "window", "model", "reference", "n_tournaments", "model_log_loss", "reference_log_loss", "model_minus_reference", "ci_lo", "ci_hi"]); w.writerows(out)

if __name__ == "__main__":
    main()
