# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Validation and monthly predictions for the smoothed archetype-pair model."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
PROC = RESULTS
from scripts.logistic_baseline import log_loss, MIN_TRAIN_MONTH_EXCLUSIVE
from scripts.unified_data import load_matches
K_GRID=[2,5,10,20,50,100]

def fit_matchup_rate(train_rows, k):
    """The specific pair's own observed win rate, pooled over both recorded
    orientations, shrunk toward 0.5 with pseudo-count k. Mirror matches are
    hard-coded to exactly 0.5."""
    pair_wins = defaultdict(lambda: defaultdict(float))
    pair_total = defaultdict(float)
    for r in train_rows:
        a, b, y = r["archetype_A"], r["archetype_B"], r["y"]
        key = frozenset((a, b))
        pair_total[key] += 1
        winner_arch = a if y == 1 else b
        pair_wins[key][winner_arch] += 1

    def predict(a, b):
        if a == b:
            return 0.5
        key = frozenset((a, b))
        n = pair_total.get(key, 0)
        w = pair_wins[key].get(a, 0)
        return (w + k * 0.5) / (n + k)

    return predict


def tune_k(rows, fit_fn, label):
    test_months_all = sorted(set(r["month"] for r in rows if r["month"] >= MIN_TRAIN_MONTH_EXCLUSIVE))
    val_months = test_months_all[:3]
    assert val_months == ["2024-04", "2024-05", "2024-06"], val_months
    tune_train = [r for r in rows if r["month"] < MIN_TRAIN_MONTH_EXCLUSIVE]
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["month"]].append(r)
    val_rows = [r for m in val_months for r in by_month[m]]

    print(f"Tuning {label}'s k over {K_GRID} using tune set = training-up-to-2024-04 -> validated on {val_months}")
    best_k, best_loss = None, float("inf")
    for k in K_GRID:
        predict = fit_fn(tune_train, k)
        losses = [log_loss(r["y"], predict(r["archetype_A"], r["archetype_B"])) for r in val_rows]
        mean_loss = sum(losses) / len(losses)
        print(f"  k={k}: mean logloss over {val_months} = {mean_loss:.5f}")
        if mean_loss < best_loss:
            best_loss, best_k = mean_loss, k
    print(f"Selected k={best_k} for {label}\n")
    return best_k


def main():
    rows=load_matches(["tournament_id","match_id","archetype_A","archetype_B","month","y"])
    k=tune_k(rows,fit_matchup_rate,"archetype-pair")
    assert k==50, "Selected pair smoothing differs from paper"
    by_month=defaultdict(list)
    for r in rows: by_month[r["month"]].append(r)
    train=[r for r in rows if r["month"] < "2024-04"]
    PROC.mkdir(parents=True,exist_ok=True)
    with (PROC/"FINAL_cardonly_vs_playerskill_predictions.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["tournament_id","month","match_id","observed_y","cardonly_prob","cardonly_logloss"])
        for month in sorted(m for m in by_month if m >= "2024-04"):
            test=by_month[month]
            if month >= "2024-07":
                predict=fit_matchup_rate(train,k)
                for r in test:
                    p=predict(r["archetype_A"],r["archetype_B"])
                    w.writerow([r["tournament_id"],month,r["match_id"],r["y"],round(p,6),round(log_loss(r["y"],p),6)])
                print(month,len(test),flush=True)
            train=train+test

if __name__=="__main__": main()
