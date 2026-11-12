# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Compare tournament and monthly updating of archetype-pair counts.
Historical evaluation only, with a paired four-month block bootstrap."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.rating_benchmarks import setup, PROC  # noqa: E402
from scripts.match_data import pair_counts, card_from_counts  # noqa: E402
from scripts.score_utils import block_bootstrap, tournament_stats  # noqa: E402

D, order, groups, *_ , month_eff = setup()
n_ar = D.n_ar; N = len(D.y); npair = np.zeros(n_ar * n_ar); wpair = np.zeros(n_ar * n_ar); p = np.zeros(N)
for t in order:
    ix = groups[t]; p[ix] = card_from_counts(D.aa[ix], D.ab[ix], npair, wpair, n_ar)
    if np.any(ix >= D.n_train):
        assert np.all(ix >= D.n_train)
        continue  # October remains frozen, although only historical scores are reported.
    n_m, w_m = pair_counts(D.aa[ix], D.ab[ix], D.y[ix], n_ar); npair += n_m; wpair += w_m
stored = {r["match_id"]: (float(r["cardonly_prob"]), float(r["cardonly_logloss"])) for r in csv.DictReader(open(PROC / "FINAL_cardonly_vs_playerskill_predictions.csv"))}
idx = [i for i in range(D.n_train) if D.mid[i] in stored]
y = D.y[idx]; pf = np.clip(p[idx], 1e-12, 1 - 1e-12); ll = -(y * np.log(pf) + (1 - y) * np.log(1 - pf))
conv = [{"tournament_id": D.tid[i], "month": month_eff[i], "match_id": D.mid[i], "hps_logloss": float(l)} for i, l in zip(idx, ll)]
ref = {m: v[1] for m, v in stored.items()}
st = tournament_stats(conv, ref); obs, lo, hi = block_bootstrap(st, lambda s: s[1] - s[2])
print(f"card-only monthly {np.mean([s[1] for s in st]):.6f} -> tournament-fresh {np.mean([s[2] for s in st]):.6f}; gain {obs:+.6f} [{lo:+.6f}, {hi:+.6f}]; n={len(st)}")
with open(PROC / "FINAL_cardonly_fresh_check.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n_tournaments", "monthly_log_loss", "fresh_log_loss", "gain", "ci_lo", "ci_hi"])
    w.writerow([len(st), round(float(np.mean([s[1] for s in st])), 6), round(float(np.mean([s[2] for s in st])), 6), round(obs, 6), round(lo, 6), round(hi, 6)])
