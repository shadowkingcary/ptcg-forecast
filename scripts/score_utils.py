# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Shared tournament means and paired historical block bootstrap."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, random, statistics
from collections import defaultdict
SEED=20261011
N_REPLICATES=5000
BLOCK_SIZE=4

def load_ll(path, col):
    return {r["match_id"]: float(r[col]) for r in csv.DictReader(open(path))}


def tournament_stats(hps_rows, ref_ll):
    """Per-tournament (month, mean ref loss - mean hps loss) and own mean loss."""
    by_t = defaultdict(list)
    for r in hps_rows:
        by_t[r["tournament_id"]].append((r["month"], ref_ll[r["match_id"]], float(r["hps_logloss"])))
    return [(v[0][0], statistics.mean(x[1] for x in v), statistics.mean(x[2] for x in v))
            for v in by_t.values()]


def block_bootstrap(stats, idx_fn):
    """stats: list of (month, ref_mean, hps_mean). idx_fn -> per-tournament scalar."""
    by_month = defaultdict(list)
    for s in stats:
        by_month[s[0]].append(idx_fn(s))
    months = sorted(by_month)
    n_m = len(months)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_REPLICATES):
        res = []
        for _ in range(-(-n_m // BLOCK_SIZE)):
            st = rng.randint(0, n_m - BLOCK_SIZE)
            for m in months[st:st + BLOCK_SIZE]:
                res.extend(by_month[m])
        boots.append(statistics.mean(res))
    boots.sort()
    obs = statistics.mean(v for m in months for v in by_month[m])
    return obs, boots[int(0.025 * N_REPLICATES)], boots[int(0.975 * N_REPLICATES)]
