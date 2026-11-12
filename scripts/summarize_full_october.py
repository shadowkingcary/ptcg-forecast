# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Recompute manuscript summaries from aligned September-cutoff prediction files.

Produces one JSON ledger for scores, paired intervals, calibration, sharpness,
and October comparisons. Four-month historical bootstrap blocks and
independent-tournament October bootstrap draws use the existing seed/replicates.
No model is fit and no hyperparameter is selected here.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
from summarize_dataset import build_summary

ROOT = Path(__file__).resolve().parents[1]
P = RESULTS
OUT = P / "full_october_2026_run"
SEED, B = 20261011, 5000
SOURCES = {
    "M1": ("FINAL_hier_full_predictions.csv", "m1_prob", "m1_logloss"),
    "Card-only": ("FINAL_cardonly_vs_playerskill_predictions.csv", "cardonly_prob", "cardonly_logloss"),
    "Player-Skill": ("FINAL_playerskill_pooled_predictions.csv", "pooled_prob", "pooled_logloss"),
    "Overall": ("FINAL_playerskill_generalonly_predictions.csv", "prob", "logloss"),
    "Joint": ("FINAL_joint_predictions.csv", "prob", "logloss"),
}


def probability_metrics(p, y):
    bins = np.minimum((p * 10).astype(int), 9)
    n = np.bincount(bins, minlength=10)
    sp = np.bincount(bins, weights=p, minlength=10)
    sy = np.bincount(bins, weights=y, minlength=10)
    occupied = n > 0
    rmse = math.sqrt(float(np.sum((sp[occupied] - sy[occupied]) ** 2 / n[occupied]) / len(p)))
    broad = np.digitize(p, [.3, .4, .5, .6, .7])
    return {"mad": float(np.abs(p - .5).mean()), "outside_04_06": float(((p < .4) | (p > .6)).mean()),
            "outside_03_07": float(((p < .3) | (p > .7)).mean()), "calibration_rmse": rmse,
            "brier": float(((p - y) ** 2).mean()), "accuracy_all": float(((p > .5) == y).mean()),
            "bins": [{"bin": int(i), "n": int(n[i]), "mean_pred": float(sp[i] / n[i]), "observed": float(sy[i] / n[i])} for i in np.where(occupied)[0]],
            "broad_bins": [{"bin": int(i), "n": int((broad == i).sum()), "mean_pred": float(p[broad == i].mean()),
                            "observed": float(y[broad == i].mean())} for i in np.unique(broad)]}


def bootstrap(values, months):
    unique = sorted(set(months))
    totals = np.array([values[np.array(months) == m].sum() for m in unique])
    counts = np.array([sum(x == m for x in months) for m in unique])
    rng = random.Random(SEED)
    draws = np.zeros((B, len(unique)), dtype=np.int16)
    for row in draws:
        for _ in range(math.ceil(len(unique) / 4)):
            start = rng.randint(0, len(unique) - 4)
            row[start:start + 4] += 1
    samples = np.sort((draws @ totals) / (draws @ counts))
    return [float(values.mean()), float(samples[125]), float(samples[4875])]


def main():
    with (P / SOURCES["M1"][0]).open() as f:
        base = list(csv.DictReader(f))
    ids = [r["match_id"] for r in base]
    assert len(ids) == len(set(ids)) == 926811
    index = {mid: i for i, mid in enumerate(ids)}
    y = np.array([int(r["observed_y"]) for r in base])
    month = np.array([r["month"] for r in base])
    tids, tc = np.unique([r["tournament_id"] for r in base], return_inverse=True)
    tmonth = np.empty(len(tids), dtype="U7")
    tmonth[tc] = month
    counts = np.bincount(tc)
    probs, losses = {}, {}
    for name, (filename, pcol, lcol) in SOURCES.items():
        p, loss, seen = np.empty(len(ids)), np.empty(len(ids)), np.zeros(len(ids), dtype=bool)
        with (P / filename).open() as stream:
            for row in csv.DictReader(stream):
                i = index[row["match_id"]]
                assert not seen[i] and int(row["observed_y"]) == y[i]
                assert row["month"] == month[i]
                p[i], loss[i], seen[i] = float(row[pcol]), float(row[lcol]), True
        assert seen.all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        probs[name], losses[name] = p, loss
    losses["Constant"] = np.full(len(ids), math.log(2))
    tloss = {name: np.bincount(tc, weights=loss) / counts for name, loss in losses.items()}
    result = {"primary": {}, "later": {}, "comparisons": {}, "sources": {}}
    for label, mask in (("primary", month >= "2024-07"), ("later", month >= "2025-01")):
        tm = tmonth >= ("2024-07" if label == "primary" else "2025-01")
        for name, p in probs.items():
            result[label][name] = {**probability_metrics(p[mask], y[mask]),
                                   "matches": int(mask.sum()), "tournaments": int(tm.sum()),
                                   "tournament_log_loss": float(tloss[name][tm].mean()),
                                   "match_log_loss": float(losses[name][mask].mean())}
        result["comparisons"][label] = {}
        for model, reference in [("M1", "Constant"), ("Card-only", "M1"),
                                  ("Overall", "Card-only"), ("Player-Skill", "Overall"),
                                  ("Player-Skill", "Card-only"), ("Joint", "Player-Skill")]:
            delta = tloss[reference][tm] - tloss[model][tm]
            result["comparisons"][label][model + " vs " + reference] = {
                "gain_ci": bootstrap(delta, tmonth[tm]), "tournaments_better": int((delta > 0).sum()),
                "fraction_tournaments_better": float((delta > 0).mean()),
                "match_weighted_gain": float((losses[reference][mask] - losses[model][mask]).mean())}
    with (P / "FINAL_october_predictions.csv").open() as f:
        hold = list(csv.DictReader(f))
    assert len(hold) == len({r["match_id"] for r in hold}) == 48288
    assert not (set(ids) & {r["match_id"] for r in hold})
    hy = np.array([int(r["observed_y"]) for r in hold])
    ht, hc = np.unique([r["tournament_id"] for r in hold], return_inverse=True)
    hn = np.bincount(hc)
    hp = {name: np.array([float(r[name]) for r in hold]) for name in hold[0] if name not in ("match_id", "tournament_id", "tournament_date", "observed_y")}
    hl = {name: -(hy * np.log(np.clip(p, 1e-12, 1)) + (1 - hy) * np.log(np.clip(1 - p, 1e-12, 1))) for name, p in hp.items()}
    htl = {name: np.bincount(hc, weights=loss) / hn for name, loss in hl.items()}
    draw = np.random.default_rng(SEED).integers(0, len(ht), (B, len(ht)))
    result["october"], result["october_comparisons"] = {}, {}
    for name, p in hp.items():
        result["october"][name] = {**probability_metrics(p, hy), "tournament_log_loss": float(htl[name].mean()), "match_log_loss": float(hl[name].mean())}
        delta = htl["Player-Skill"] - htl[name]
        boot = np.sort(delta[draw].mean(1))
        result["october_comparisons"][name + " vs Player-Skill"] = [float(delta.mean()), float(boot[125]), float(boot[4875])]
    for model, reference in (("Player-Skill", "Card-only"), ("Player-Skill", "Overall-record control"),
                             ("Joint model + recal.", "Joint model")):
        delta = htl[reference] - htl[model]; boot = np.sort(delta[draw].mean(1))
        inverse = result["october_comparisons"].get(reference + " vs " + model)
        result["october_comparisons"][model + " vs " + reference] = ([-inverse[0], -inverse[2], -inverse[1]] if inverse is not None else [float(delta.mean()), float(boot[125]), float(boot[4875])])
    result["dataset"] = build_summary()
    for filename in sorted(set(v[0] for v in SOURCES.values()) | {"FINAL_october_predictions.csv"}):
        result["sources"][filename] = hashlib.sha256((P / filename).read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved paper_results.json; primary/holdout identity checks passed.")
    for period in ("primary", "october"):
        print(period, {k: round(v["tournament_log_loss"], 6) for k, v in result[period].items()})


if __name__ == "__main__":
    main()
