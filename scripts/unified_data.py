# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""
The single, read-only entry point to the unified match tables.

Every model and analysis script gets its input from here, so there is one definition of the data, one file layout, and no script
writes to it. The tables live in data/unified/ (matches.csv.gz and matches_oct2026_holdout.csv.gz; a plain .csv of the same name is
also accepted). Results are written elsewhere (external results and figures directories); nothing in this module opens a file for writing.

Schema (one row per decisive match; all values are read as strings, as before):
    tournament_id, tournament_date, era, match_id, player_A, player_B, archetype_A, archetype_B, winner, round

    load_matches(columns)   the main table filtered to dates before October 1, 2026, sorted by tournament_date (stable)
    load_holdout(columns)   the full October 2026 retrospective evaluation, in file order

`columns` lists only what the caller uses, so a model that ignores players, for example, never loads them. Besides the schema
columns, two derived columns can be requested: "month" (tournament_date[:7]) and "y" (1 if the winner is A, else 0).
Requesting a column that is not in the schema raises an error; a script that reads a field it did not request fails on a KeyError
instead of silently using data it did not declare.

Set UNIFIED_TRACE=<file> to record, at exit, which columns of the loaded rows a script actually accessed (used to check the
column lists in the scripts).
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external

import atexit
import csv
import gzip
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "unified"
MATCHES = "matches.csv"
HOLDOUT = "matches_oct2026_holdout.csv"
TRAIN_END = "2026-10-01"  # exclusive: no October outcomes in the historical fits
HOLDOUT_MONTH = "2026-10"
HOLDOUT_END = "2026-11-01"

SCHEMA = ["tournament_id", "tournament_date", "era", "match_id", "player_A", "player_B", "archetype_A",
          "archetype_B", "winner", "round"]
DERIVED = ["month", "y"]

_ACCESSED = set()


class _TracedRow(dict):
    def __getitem__(self, k):
        _ACCESSED.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        _ACCESSED.add(k)
        return super().get(k, default)


def _dump_trace():
    path = os.environ.get("UNIFIED_TRACE")
    if path and _ACCESSED:
        with open(path, "a") as f:
            f.write(",".join(sorted(_ACCESSED)) + "\n")


atexit.register(_dump_trace)


def _resolve(name, path=None):
    if path is not None:
        return Path(path)
    for cand in (DATA_DIR / (name + ".gz"), DATA_DIR / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name}(.gz) not found in {DATA_DIR}")


def _open_text(path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if str(path).endswith(".gz") else open(path, newline="", encoding="utf-8")


def _read(name, columns, sort, path=None, before=None, period=None):
    columns = list(SCHEMA + DERIVED if columns is None else columns)
    unknown = [c for c in columns if c not in SCHEMA and c not in DERIVED]
    if unknown:
        raise KeyError(f"unknown column(s) {unknown}; schema is {SCHEMA} plus derived {DERIVED}")
    need = {c for c in columns if c in SCHEMA}
    if "month" in columns:
        need.add("tournament_date")
    if "y" in columns:
        need.add("winner")
    if sort:
        need.add("tournament_date")
    if before or period:
        need.add("tournament_date")
    rows = []
    with _open_text(_resolve(name, path)) as f:
        reader = csv.DictReader(f)
        missing = [c for c in need if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"column(s) {missing} missing from {name}; file has {reader.fieldnames}")
        trace = bool(os.environ.get("UNIFIED_TRACE"))
        for rec in reader:
            date = rec.get("tournament_date", "")
            if before and date >= before:
                continue
            if period and not period[0] <= date < period[1]:
                raise ValueError(f"Date {date!r} lies outside the October evaluation period")
            r = {c: rec[c] for c in need}
            if "month" in columns:
                r["month"] = rec["tournament_date"][:7]
            if "y" in columns:
                r["y"] = 1 if rec["winner"] == "A" else 0
            rows.append(r)
    if sort:
        rows.sort(key=lambda r: r["tournament_date"])
    keep = set(columns)
    out = []
    for r in rows:
        r = {c: r[c] for c in r if c in keep}
        out.append(_TracedRow(r) if trace else r)
    return out


def load_matches(columns=None, path=None, sort=True):
    """History strictly before October 2026; an explicit path uses the same cutoff."""
    return _read(MATCHES, columns, sort=sort, path=path, before=TRAIN_END)


def load_holdout(columns=None, path=None):
    """Full October 2026 retrospective evaluation, in file order."""
    return _read(HOLDOUT, columns, sort=False, path=path, period=(TRAIN_END, HOLDOUT_END))
