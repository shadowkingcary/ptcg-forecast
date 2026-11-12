# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Control for the rating benchmarks: Player-Skill with ONLY the player's overall record
(no player-archetype component; k_s -> infinity, k_o = 20 frozen) on top of the same card estimate
(k_p = 50). Same expanding-window monthly protocol and eligible matches as every other model.
Outputs: the external results directory, FINAL_playerskill_generalonly_predictions.csv (July 2024 onward) and
FINAL_playerskill_generalonly_holdout_predictions.csv (Oct 1-31 2026; training = months < 2026-10)."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.logistic_baseline import log_loss  # noqa: E402
from scripts.archetype_pair import fit_matchup_rate  # noqa: E402
from scripts.player_records import fit_player_skill_pooled, logit, sigmoid  # noqa: E402
from scripts.unified_data import load_matches, load_holdout  # noqa: E402
COLUMNS = ["tournament_id", "match_id", "player_A", "player_B", "archetype_A", "archetype_B", "winner", "month", "y"]  # the unified-table columns this script reads

PROC = RESULTS
K_O, K_S_INF, K_P = 20, 1e9, 50


def predict_rows(train, rows):
    card = fit_matchup_rate(train, K_P); skill = fit_player_skill_pooled(train, K_O, K_S_INF)
    return [sigmoid(logit(card(r["archetype_A"], r["archetype_B"])) + skill(r["player_A"], r["archetype_A"]) - skill(r["player_B"], r["archetype_B"])) for r in rows]


def main():
    rows = load_matches(COLUMNS)
    by_m = defaultdict(list)
    for r in rows: by_m[r["month"]].append(r)
    months = sorted(by_m); train = [r for r in rows if r["month"] < "2024-04"]; out = []
    for m in [x for x in months if x >= "2024-04"]:
        if m >= "2024-07":
            for r, p in zip(by_m[m], predict_rows(train, by_m[m])):
                out.append({"tournament_id": r["tournament_id"], "month": m, "match_id": r["match_id"], "observed_y": r["y"],
                            "prob": round(p, 6), "logloss": round(log_loss(r["y"], p), 6)})
        train = train + by_m[m]
        print(m, flush=True)
    hold = load_holdout(COLUMNS)
    ph = predict_rows(train, hold)
    with open(PROC / "FINAL_playerskill_generalonly_holdout_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["match_id", "tournament_id", "observed_y", "prob"])
        for r, p in zip(hold, ph): w.writerow([r["match_id"], r["tournament_id"], r["y"], round(p, 6)])
    with open(PROC / "FINAL_playerskill_generalonly_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    main()
