# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Monthly expanding-window logistic baseline, with unchanged C=1 fitting."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import csv, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
PROC = RESULTS
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from scripts.unified_data import load_matches
MIN_TRAIN_MONTH_EXCLUSIVE="2024-04"

def fit_m1(train_rows, weights=None, C=1.0):
    """L2 logistic regression with focal and opponent archetype indicators."""
    archetypes = sorted(set(r["archetype_A"] for r in train_rows) | set(r["archetype_B"] for r in train_rows))
    idx = {a: i for i, a in enumerate(archetypes)}
    n_arch = len(archetypes)

    n = len(train_rows)
    rows_idx, cols_idx = [], []
    for i, r in enumerate(train_rows):
        rows_idx.append(i)
        cols_idx.append(idx[r["archetype_A"]])
        rows_idx.append(i)
        cols_idx.append(n_arch + idx[r["archetype_B"]])
    X = sparse.csr_matrix((np.ones(len(rows_idx)), (rows_idx, cols_idx)), shape=(n, 2 * n_arch))
    y = np.array([r["y"] for r in train_rows])

    model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=200)
    model.fit(X, y, sample_weight=weights)

    def predict_batch(test_rows):
        n_t = len(test_rows)
        r_idx, c_idx = [], []
        for i, r in enumerate(test_rows):
            a, b = r["archetype_A"], r["archetype_B"]
            if a in idx:
                r_idx.append(i)
                c_idx.append(idx[a])
            if b in idx:
                r_idx.append(i)
                c_idx.append(n_arch + idx[b])
        Xt = sparse.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(n_t, 2 * n_arch))
        return model.predict_proba(Xt)[:, 1]

    return predict_batch


def log_loss(y_true, p):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return -(y_true * math.log(p) + (1 - y_true) * math.log(1 - p))


def brier(y_true, p):
    return (p - y_true) ** 2


def main():
    rows=load_matches(["tournament_id","match_id","archetype_A","archetype_B","month","y"])
    by_month=defaultdict(list)
    for r in rows: by_month[r["month"]].append(r)
    train=[r for r in rows if r["month"] < MIN_TRAIN_MONTH_EXCLUSIVE]
    PROC.mkdir(parents=True,exist_ok=True)
    path=PROC/"FINAL_hier_full_predictions.csv"
    with path.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["tournament_id","month","match_id","observed_y","m1_prob","m1_logloss"])
        for month in sorted(m for m in by_month if m >= MIN_TRAIN_MONTH_EXCLUSIVE):
            test=by_month[month]
            if month >= "2024-07":
                probs=fit_m1(train)(test)
                for r,p in zip(test,probs):
                    w.writerow([r["tournament_id"],month,r["match_id"],r["y"],round(float(p),6),round(log_loss(r["y"],float(p)),6)])
                print(month, len(test),flush=True)
            train=train+test

if __name__=="__main__": main()
