# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Full October 2026 retrospective validation with all fits frozen before October 1.

Previously selected parameters are retained. October 1-10 informed earlier
model development, so this is not a wholly unseen confirmatory holdout.
Tournament-bootstrap intervals are indicative and omit shared-model and
selection uncertainty. Match-level predictions are retained for audit.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.logistic_baseline import fit_m1  # noqa: E402
from scripts.archetype_pair import fit_matchup_rate  # noqa: E402
from scripts.player_records import fit_player_skill_pooled, logit, sigmoid  # noqa: E402
from scripts.match_data import prep, slicer  # noqa: E402
from scripts.joint_hierarchical_model import fit_predict as joint_predict  # noqa: E402
from scripts.posthoc_calibration import fit_ab, lg, SRC, FIT_MONTHS  # noqa: E402
from scripts.unified_data import load_matches, load_holdout  # noqa: E402
COLUMNS = ["tournament_id", "match_id", "tournament_date", "player_A", "player_B", "archetype_A", "archetype_B", "winner", "month", "y"]  # the unified-table columns this script reads

ROOT = Path(__file__).resolve().parent.parent
PROC = RESULTS
SEED, B = 20261011, 5000


def main():
    train_rows = [r for r in load_matches(COLUMNS) if r["month"] < "2026-10"]
    hold = load_holdout(COLUMNS)
    for r in hold:
        r["y"] = 1 if r["winner"] == "A" else 0
        r["month"] = "2026-10"
    y = np.array([r["y"] for r in hold], float)
    tid = np.array([r["tournament_id"] for r in hold])
    preds = {}

    p_m1 = fit_m1(train_rows)(hold); preds["M1"] = np.asarray(p_m1, float)
    skill = fit_player_skill_pooled(train_rows, 20, 50)
    card = fit_matchup_rate(train_rows, 50)
    def ps(cardf):
        return np.array([sigmoid(logit(cardf(r["archetype_A"], r["archetype_B"])) + skill(r["player_A"], r["archetype_A"])
                                 - skill(r["player_B"], r["archetype_B"])) for r in hold])
    preds["Card-only"] = np.array([card(r["archetype_A"], r["archetype_B"]) for r in hold])
    preds["Player-Skill"] = ps(card)
    overall = fit_player_skill_pooled(train_rows, 20, 1e9)
    preds["Overall-record control"] = np.array([sigmoid(logit(card(r["archetype_A"], r["archetype_B"])) + overall(r["player_A"], r["archetype_A"]) - overall(r["player_B"], r["archetype_B"])) for r in hold])

    allrows = train_rows + hold
    d = prep(allrows); is_h = np.arange(len(allrows)) >= len(train_rows)
    tr, te = slicer(d, ~is_h), slicer(d, is_h)
    preds["Joint model"] = joint_predict(tr, te, d["n_ar"], (0.28, 0.45, 0.14))

    # frozen calibration parameters from Jul-Dec 2024 predictions
    for name, key in (("Joint model", "Joint model"),):
        fn, col = SRC[key]
        rr = [r for r in csv.DictReader(open(PROC / fn)) if r["month"] in FIT_MONTHS]
        a, b = fit_ab(lg(np.array([float(r[col]) for r in rr])), np.array([int(r["observed_y"]) for r in rr], float))
        preds[name + " + recal."] = 1 / (1 + np.exp(-(a + b * lg(preds[name]))))

    with (PROC / "FINAL_october_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "tournament_id", "tournament_date", "observed_y", *preds])
        for i, r in enumerate(hold):
            writer.writerow([r["match_id"], r["tournament_id"], r["tournament_date"], int(y[i]), *[float(p[i]) for p in preds.values()]])

    def tloss(p):
        p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12); l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.array([l[tid == t].mean() for t in np.unique(tid)])
    def rmse(p):
        p = np.asarray(p, float); bn = np.minimum((p * 10).astype(int), 9)
        return math.sqrt(sum((bn == k).sum() * (p[bn == k].mean() - y[bn == k].mean()) ** 2 for k in np.unique(bn)) / len(p))
    L = {k: tloss(v) for k, v in preds.items()}
    ref = L["Player-Skill"]; rng = np.random.default_rng(SEED); nt = len(ref)
    idx = rng.integers(0, nt, size=(B, nt))
    print(f"holdout: {len(hold)} matches, {nt} tournaments")
    out = []
    for k, l in L.items():
        diff = ref - l; bs = np.sort(diff[idx].mean(1))
        row = [k, round(l.mean(), 6), round(diff.mean(), 6), round(bs[int(.025 * B)], 6), round(bs[int(.975 * B)], 6), round(rmse(preds[k]), 6)]
        out.append(row); print(row)
    with open(PROC / "FINAL_holdout_new_methods.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "tournament_log_loss", "improvement_over_player_skill", "ci_lo_naive", "ci_hi_naive", "calibration_rmse"]); w.writerows(out)

if __name__ == "__main__":
    main()
