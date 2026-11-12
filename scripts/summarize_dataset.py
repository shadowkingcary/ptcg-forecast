# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Regenerate all manuscript population counts from the two active input tables."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_summary():
    groups = {}
    for name in ("history", "october", "combined", "primary", "later", "history_mirrors"):
        groups[name] = dict(n=0, a=0, other=0, players=set(), archetypes=set(),
                            tournaments=set(), min_date="9999", max_date="")
    for filename, is_history in (("matches.csv.gz", True), ("matches_oct2026_holdout.csv.gz", False)):
        with gzip.open(ROOT / "data/unified" / filename, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                date = row["tournament_date"]
                if is_history and date >= "2026-10-01":
                    continue
                if not is_history:
                    assert "2026-10-01" <= date < "2026-11-01"
                names = ["combined", "history" if is_history else "october"]
                if is_history:
                    if date >= "2024-07-01": names.append("primary")
                    if date >= "2025-01-01": names.append("later")
                    if row["archetype_A"] == row["archetype_B"]: names.append("history_mirrors")
                for name in names:
                    g = groups[name]
                    g["n"] += 1
                    g["a"] += row["winner"] == "A"
                    g["other"] += (row["archetype_A"] == "other") + (row["archetype_B"] == "other")
                    g["players"].update((row["player_A"], row["player_B"]))
                    g["archetypes"].update((row["archetype_A"], row["archetype_B"]))
                    g["tournaments"].add(row["tournament_id"])
                    g["min_date"] = min(date, g["min_date"])
                    g["max_date"] = max(date, g["max_date"])
    assert not groups["history"]["tournaments"] & groups["october"]["tournaments"]
    summary = {name: dict(matches=g["n"], tournaments=len(g["tournaments"]),
                         players=len(g["players"]), archetypes=len(g["archetypes"]),
                         min_date=g["min_date"], max_date=g["max_date"],
                         a_win_rate=g["a"] / g["n"], other_appearance_fraction=g["other"] / (2 * g["n"]))
               for name, g in groups.items()}
    path = RESULTS/"full_october_2026_run/dataset_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_summary(), indent=2))
