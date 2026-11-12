# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Generate the manuscript's six result tables from the full-October result ledger.

Writes Markdown table fragments and their source hashes under the run directory.
It does not edit the manuscript; prose interpretation remains an editorial step.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.journal_comparisons import holdout_comparisons

ROOT = Path(__file__).resolve().parents[1]
P = RESULTS
RUN = P / "full_october_2026_run"
R = json.loads((RUN / "paper_results.json").read_text())
T = {}
SOURCE_PATHS = {RUN / "paper_results.json"}


def f(value, digits=6, signed=False):
    value = float(value)
    if round(value, digits) == 0:
        value = 0.0  # Do not print a negative zero for floating-point residuals.
    return format(value, ("+" if signed else "") + f".{digits}f").replace("-", "−")


def ci(v, signed=False):
    return f"{f(v[0], signed=signed)} [{f(v[1])}, {f(v[2])}]"


def read(name):
    SOURCE_PATHS.add(P / name)
    with (P / name).open() as stream:
        return list(csv.DictReader(stream))


def pick(rows, **keys):
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(matches) == 1, keys
    return matches[0]


def delta(row, key="model_minus_reference", lo="ci_lo", hi="ci_hi"):
    return [float(row[k]) for k in (key, lo, hi)]


def table(key, headings, rows):
    T[key] = "\n".join(["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings),
                         *["| " + " | ".join(map(str, row)) + " |" for row in rows]])
    if not key.startswith("S"):
        # Preserve saved analysis identifiers; use record-based names in the paper.
        T[key] = T[key].replace("Player-Skill", "Archetype-specific records")


def main():
    rows = []
    labels = {"Constant": "Constant 0.5", "M1": "Logistic baseline", "Card-only": "Archetype-pair", "Overall": "Overall records", "Player-Skill": "Player-Skill", "Joint": "Joint model"}
    for model, ref in [("M1", "Constant"), ("Card-only", "M1"), ("Overall", "Card-only"), ("Player-Skill", "Overall"), ("Joint", "Player-Skill")]:
        rows.append([labels[model], f(R["primary"][model]["tournament_log_loss"]), labels[ref],
                     ci(R["comparisons"]["primary"][model + " vs " + ref]["gain_ci"])])
    table("1", ["Model", "Log loss", "Reference", "Gain [95% CI]"], rows)
    holdout = holdout_comparisons()
    SOURCE_PATHS.update({P/'FINAL_october_predictions.csv', P/'history_depth/summary.json',
                         ROOT/'scripts/journal_comparisons.py', ROOT/'scripts/history_depth_analysis.py'})
    hold_labels = {**labels, 'Overall-record control':'Overall records', 'Joint model':'Joint model'}
    sequence = [('M1','Constant'), ('Card-only','M1'), ('Overall-record control','Card-only'),
                ('Player-Skill','Overall-record control'), ('Joint model','Player-Skill')]
    table("2", ["Model", "Log loss", "Reference", "Gain [indicative 95% CI]", "Calibration RMSE"],
          [[hold_labels[model], f(R['october'][model]['tournament_log_loss']), hold_labels[ref],
            ci(holdout[model+' vs '+ref]), f(R['october'][model]['calibration_rmse'])]
           for model, ref in sequence])
    fresh = read("FINAL_rating_fresh_comparison.csv")
    card_fresh = pick(read("FINAL_cardonly_fresh_check.csv"), n_tournaments="5836")
    ps_fresh = pick(fresh, spec="primary", model="Player-Skill (tournament-fresh)")
    ps_delta = delta(ps_fresh)
    table("6", ["Model", "Monthly updates", "Tournament updates", "Gain [95% CI]"], [
        ["Archetype-pair", f(card_fresh["monthly_log_loss"]), f(card_fresh["fresh_log_loss"]), ci(delta(card_fresh, "gain"))],
        ["Player-Skill", f(ps_fresh["reference_log_loss"]), f(ps_fresh["model_log_loss"]), ci([-ps_delta[0], -ps_delta[2], -ps_delta[1]])]])
    table("4", ["Model", "Tournament-weighted log loss", "Match-weighted log loss", "Match-weighted Brier score", "Calibration RMSE"],
          [[labels[name], *[f(R["primary"][name][key]) for key in ["tournament_log_loss", "match_log_loss", "brier", "calibration_rmse"]]]
           for name in ["M1", "Card-only", "Overall", "Player-Skill", "Joint"]])
    history_path = P / 'history_depth/summary.json'
    SOURCE_PATHS.add(history_path)
    history = json.loads(history_path.read_text())
    assert history['validation']['all_sample_scores_and_intervals_match_existing_ledger']
    def history_gain(row, contrast):
        c = row['comparisons'][contrast]
        return ci([c['gain'], *c['ci']]) if c['ci'] is not None else f(c['gain']) + ' [unavailable]'
    # Keep both evaluation periods visible in the self-contained main paper.
    table('3', ['Period', 'Minimum prior matches', 'Matches (%)', 'Tournaments',
                'Overall vs pair [95% CI]', 'Player-Skill vs overall [95% CI]'],
          [[label, r['group'], f"{r['matches']:,} ({100*r['match_share']:.1f}%)", f"{r['tournaments']:,}",
            history_gain(r, 'Overall vs pair'), history_gain(r, 'Player-Skill vs overall')]
           for period, label in [('primary','Historical'),('october','Holdout')]
           for r in history['periods'][period]['groups'] if r['kind']=='depth'])
    recal = read("FINAL_posthoc_calibration.csv")
    uncal = pick(recal, model='Joint model', calibration='uncalibrated')
    cal = pick(recal, model='Joint model', calibration='frozen')
    table('5', ['Period', 'Original log loss', 'Recalibrated log loss', 'Gain [95% CI]',
                'Calibration RMSE: original / recalibrated'], [
        ['Later historical', f(uncal['tournament_log_loss']), f(cal['tournament_log_loss']),
         ci(delta(cal, 'gain_vs_uncalibrated')), f(uncal['calibration_rmse'])+' / '+f(cal['calibration_rmse'])],
        ['Holdout', f(R['october']['Joint model']['tournament_log_loss']),
         f(R['october']['Joint model + recal.']['tournament_log_loss']),
         ci(R['october_comparisons']['Joint model + recal. vs Joint model']),
         f(R['october']['Joint model']['calibration_rmse'])+' / '+f(R['october']['Joint model + recal.']['calibration_rmse'])]])
    (RUN / "manuscript_tables.json").write_text(json.dumps(T, indent=2, ensure_ascii=False) + "\n")
    hashes = {artifact_label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE_PATHS)}
    (RUN / "table_sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Generated {len(T)} manuscript result tables")


if __name__ == "__main__":
    main()
