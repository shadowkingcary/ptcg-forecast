# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Check that the vectorized count implementation with monthly updates
reproduces saved archetype-specific forecasts to six-decimal rounding precision."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.rating_benchmarks import setup, sig, PROC  # noqa: E402
from scripts.match_data import pair_counts, card_from_counts  # noqa: E402

D, *_ = setup(); n_ar = D.n_ar; N = len(D.y)
npair = np.zeros(n_ar * n_ar); wpair = np.zeros(n_ar * n_ar)
ot = np.zeros(D.n_pl); ow = np.zeros(D.n_pl); at = np.zeros(D.n_keys); aw = np.zeros(D.n_keys); p = np.zeros(N)
for m in sorted(set(D.month[:D.n_train])):
    ix = np.where(D.month == m)[0]; pa, pb, y = D.pa[ix], D.pb[ix], D.y[ix]; kA, kB = D.kA[ix], D.kB[ix]
    card = card_from_counts(D.aa[ix], D.ab[ix], npair, wpair, n_ar)
    def skill(pl, k):
        o = (ow[pl] + 10.0) / (ot[pl] + 20.0); q = np.clip((aw[k] + 50.0 * o) / (at[k] + 50.0), 1e-9, 1 - 1e-9); return np.log(q / (1 - q))
    p[ix] = sig(np.log(card / (1 - card)) + skill(pa, kA) - skill(pb, kB))
    np.add.at(ot, pa, 1); np.add.at(ot, pb, 1); np.add.at(ow, pa, y); np.add.at(ow, pb, 1 - y)
    np.add.at(at, kA, 1); np.add.at(at, kB, 1); np.add.at(aw, kA, y); np.add.at(aw, kB, 1 - y)
    n_m, w_m = pair_counts(D.aa[ix], D.ab[ix], y, n_ar); npair += n_m; wpair += w_m
ref = {r["match_id"]: float(r["pooled_prob"]) for r in csv.DictReader(open(PROC / "FINAL_playerskill_pooled_predictions.csv"))}
d = [abs(p[i] - ref[D.mid[i]]) for i in range(D.n_train) if D.mid[i] in ref]
print("matches compared:", len(d), "max |diff|:", round(max(d), 7))

assert len(d)==926811 and max(d)<=5.000001e-7, "Monthly implementation differs from saved forecasts"
