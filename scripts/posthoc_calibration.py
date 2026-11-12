# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Frozen logistic recalibration of the joint model: fit July–December 2024
forecasts, evaluate January 2025–September 2026. The holdout script applies the
same coefficients to October. Report paired loss gains and binned calibration RMSE."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, statistics, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.score_utils import block_bootstrap, tournament_stats  # noqa: E402

P = RESULTS
SRC = {
    "Joint model": ("FINAL_joint_predictions.csv", "prob"),
}
FIT_MONTHS = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))

def fit_ab(z, y):
    th = np.array([0.0, 1.0])
    for _ in range(25):  # Newton for 2-parameter logistic regression
        eta = th[0] + th[1] * z; p = 1 / (1 + np.exp(-eta))
        g = np.array([np.sum(y - p), np.sum((y - p) * z)])
        W = p * (1 - p)
        H = np.array([[W.sum(), (W * z).sum()], [(W * z).sum(), (W * z * z).sum()]])
        th = th + np.linalg.solve(H + 1e-9 * np.eye(2), g)
    return th

def ll(p, y): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def rmse(p, y):
    b = np.minimum((p * 10).astype(int), 9); s = 0.0
    for k in np.unique(b):
        m = b == k; s += m.sum() * (p[m].mean() - y[m].mean()) ** 2
    return math.sqrt(s / len(p))

def main():
    out = []
    for name, (fn, col) in SRC.items():
        rows = list(csv.DictReader(open(P / fn)))
        month = np.array([r["month"] for r in rows]); y = np.array([int(r["observed_y"]) for r in rows], float)
        p = np.array([float(r[col]) for r in rows]); z = lg(p)
        months = sorted(set(month)); test_months = [m for m in months if m > FIT_MONTHS[-1]]
        mask_fit = np.isin(month, FIT_MONTHS)
        a, b = fit_ab(z[mask_fit], y[mask_fit])
        pc_frozen = np.where(np.isin(month, test_months), 1 / (1 + np.exp(-(a + b * z))), np.nan)
        ev = np.isin(month, test_months)
        print(f"{name}: frozen a={a:.4f} b={b:.4f} (b<1 shrinks forecasts toward 0.5)")
        base_ll = ll(p, y)
        for label, pc in [("uncalibrated", p), ("frozen", pc_frozen)]:
            lloss = ll(pc[ev], y[ev])
            conv_rows = [{"tournament_id": rows[i]["tournament_id"], "month": month[i], "match_id": rows[i]["match_id"],
                          "hps_logloss": lloss_i} for i, lloss_i in zip(np.where(ev)[0], lloss)]
            ref = {rows[i]["match_id"]: base_ll[i] for i in np.where(ev)[0]}
            stats = tournament_stats(conv_rows, ref)
            teq = statistics.mean(s[2] for s in stats)
            obs, lo, hi = block_bootstrap(stats, lambda s: s[1] - s[2])
            r_ = rmse(pc[ev], y[ev])
            print(f"  {label:13s} log loss {teq:.6f}  gain vs uncal {obs:+.6f} [{lo:+.6f},{hi:+.6f}]  RMSE {r_:.6f}  n_tourn={len(stats)}")
            out.append([name, label, round(a, 4) if label == "frozen" else "", round(b, 4) if label == "frozen" else "",
                        len(stats), round(teq, 6), round(obs, 6), round(lo, 6), round(hi, 6), round(r_, 6)])
    with open(P / "FINAL_posthoc_calibration.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "calibration", "a", "b", "n_tournaments", "tournament_log_loss",
                                       "gain_vs_uncalibrated", "ci_lo", "ci_hi", "calibration_rmse"]); w.writerows(out)

if __name__ == "__main__":
    main()
