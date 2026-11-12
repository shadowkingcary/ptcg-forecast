# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Archetype-specific player records smoothed toward overall records."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
PROC = RESULTS
from scripts.logistic_baseline import log_loss, MIN_TRAIN_MONTH_EXCLUSIVE
from scripts.archetype_pair import fit_matchup_rate
from scripts.unified_data import load_matches
K_P_FROZEN=50
TUNING_MONTHS={"2024-04","2024-05","2024-06"}
POOLED_GRID=[(ko,ks) for ko in [10,20,50,100] for ks in [5,10,20,50]]
COLUMNS=["tournament_id","match_id","player_A","player_B","archetype_A","archetype_B","month","y"]

def make_predict(card_predict, skill):
    def predict(pa, aa, pb, ab):
        z = logit(card_predict(aa, ab)) + skill(pa, aa) - skill(pb, ab)
        return sigmoid(z)
    return predict

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def logit(p, eps=1e-9):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_player_skill_pooled(train_rows, k_o, k_s):
    overall_wins = defaultdict(float)
    overall_total = defaultdict(float)
    arche_wins = defaultdict(float)
    arche_total = defaultdict(float)
    for r in train_rows:
        pa, pb = r["player_A"], r["player_B"]
        aa, ab = r["archetype_A"], r["archetype_B"]
        overall_total[pa] += 1
        overall_total[pb] += 1
        arche_total[(pa, aa)] += 1
        arche_total[(pb, ab)] += 1
        if r["y"] == 1:
            overall_wins[pa] += 1
            arche_wins[(pa, aa)] += 1
        else:
            overall_wins[pb] += 1
            arche_wins[(pb, ab)] += 1

    def skill(player, archetype):
        ot = overall_total.get(player, 0)
        ow = overall_wins.get(player, 0)
        overall_p = (ow + k_o * 0.5) / (ot + k_o)
        at = arche_total.get((player, archetype), 0)
        aw = arche_wins.get((player, archetype), 0)
        p = (aw + k_s * overall_p) / (at + k_s)
        return logit(p)

    return skill


def months_and_by_month(rows):
    months = sorted(set(r["month"] for r in rows))
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["month"]].append(r)
    return months, by_month


def val_rows_and_tune_train(rows):
    months, by_month = months_and_by_month(rows)
    tune_train = [r for r in rows if r["month"] < MIN_TRAIN_MONTH_EXCLUSIVE]
    val_months = ["2024-04", "2024-05", "2024-06"]
    val_rows = [r for m in val_months for r in by_month[m]]
    return tune_train, val_rows, val_months


def tune_pooled(rows):
    tune_train, val_rows, val_months = val_rows_and_tune_train(rows)
    card_predict = fit_matchup_rate(tune_train, K_P_FROZEN)
    print(f"Tuning pooled (k_o, k_s) over {len(POOLED_GRID)} combos, card k_p frozen at {K_P_FROZEN}")
    best_combo, best_loss = None, float("inf")
    for k_o, k_s in POOLED_GRID:
        skill = fit_player_skill_pooled(tune_train, k_o, k_s)
        predict = make_predict(card_predict, skill)
        losses = [
            log_loss(r["y"], predict(r["player_A"], r["archetype_A"], r["player_B"], r["archetype_B"]))
            for r in val_rows
        ]
        mean_loss = sum(losses) / len(losses)
        print(f"  k_o={k_o}, k_s={k_s}: mean logloss = {mean_loss:.5f}")
        if mean_loss < best_loss:
            best_loss, best_combo = mean_loss, (k_o, k_s)
    print(f"Selected (k_o, k_s)={best_combo}\n")
    return best_combo


def run_expanding_window(rows, variant, params):
    months, by_month = months_and_by_month(rows)
    test_months = [m for m in months if m >= MIN_TRAIN_MONTH_EXCLUSIVE]
    train_rows = [r for r in rows if r["month"] < MIN_TRAIN_MONTH_EXCLUSIVE]

    pred_rows = []
    for month in test_months:
        test_rows = by_month[month]
        card_predict = fit_matchup_rate(train_rows, K_P_FROZEN)
        k_o, k_s = params
        skill = fit_player_skill_pooled(train_rows, k_o, k_s)
        predict = make_predict(card_predict, skill)
        for r in test_rows:
            p = predict(r["player_A"], r["archetype_A"], r["player_B"], r["archetype_B"])
            if month not in TUNING_MONTHS:
                pred_rows.append({
                    "tournament_id": r["tournament_id"], "month": month, "match_id": r["match_id"],
                    "observed_y": r["y"], f"{variant}_prob": round(p, 6),
                    f"{variant}_logloss": round(log_loss(r["y"], p), 6),
                })
        print(f"[{variant}] month {month}: train_n={len(train_rows)} test_n={len(test_rows)}"
              f"{' (tuning month, excluded)' if month in TUNING_MONTHS else ''}")
        train_rows = train_rows + test_rows

    out_path = RESULTS / f"FINAL_playerskill_{variant}_predictions.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader(); w.writerows(pred_rows)
    print(f"Wrote {len(pred_rows)} rows -> {out_path}\n")
    return pred_rows, out_path


def main():
    rows=load_matches(COLUMNS)
    params=tune_pooled(rows)
    assert params==(20,50), "Selected player smoothing differs from paper"
    PROC.mkdir(parents=True,exist_ok=True)
    run_expanding_window(rows,"pooled",params)

if __name__=="__main__": main()
