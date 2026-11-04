"""Early experiment: blend2 plus a per-class multiplicative calibration (coordinate ascent on out-of-fold MAP@3 for the
12 most frequent classes, with a nested cross-validation check). Gain: about +0.003.

Usage:
    python scripts/experiments/blend3.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
mm = json.load(open("work/cpu_meta_qwen3.5-9b.json"))
C1 = mm["classes"]
A_oof = np.load("work/cpu_oof_qwen3.5-9b.npy")
A_te = np.load("work/cpu_test_qwen3.5-9b.npy")
bm = json.load(open("data/external/baseline_ocr/probs_meta_baseline.json"))
C2 = bm["classes"]
B_oof = np.load("data/external/baseline_ocr/probs_oof_baseline.npy")
B_te = np.load("data/external/baseline_ocr/probs_test_baseline.npy")
B2 = np.zeros_like(A_oof)
B2t = np.zeros_like(A_te)
for j, c in enumerate(C2):
    if c in C1:
        B2[:, C1.index(c)] = B_oof[:, j]
        B2t[:, C1.index(c)] = B_te[:, j]
y = tr.root_code.values
yi = np.array([C1.index(c) for c in y])


def map3(P, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    top = np.argsort(-P[idx], 1)[:, :3]
    s = 0
    for row, t in zip(top, yi[idx]):
        s += (1 / (list(row).index(t) + 1)) if t in row else 0
    return s / len(idx)


P0 = 0.4 * A_oof + 0.6 * B2
T0 = 0.4 * A_te + 0.6 * B2t
freq = pd.Series(y).value_counts()
cal_classes = [C1.index(c) for c in freq.index[:12]]
grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8]


def fit_w(idx):
    w = np.ones(len(C1))
    w[C1.index("NONE")] = 0.6
    for _ in range(3):
        for c in cal_classes:
            best = (map3(P0 * w, idx), w[c])
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                sc = map3(P0 * w2, idx)
                if sc > best[0] + 1e-9:
                    best = (sc, g)
            w[c] = best[1]
    return w


# nested check: fit weights on 4/5, evaluate on 1/5
strat = pd.Series(y).where(pd.Series(y).map(freq) >= 5, "RARE")
cv = []
base = []
for trn, va in StratifiedKFold(5, shuffle=True, random_state=0).split(P0, strat):
    w = fit_w(trn)
    cv.append(map3(P0 * w, va))
    w0 = np.ones(len(C1))
    w0[C1.index("NONE")] = 0.6
    base.append(map3(P0 * w0, va))
print("nested CV: none-scale only %.4f -> per-class calib %.4f" % (np.mean(base), np.mean(cv)))
w = fit_w(np.arange(len(y)))
print("full-fit OOF %.4f weights:" % map3(P0 * w), {C1[c]: round(w[c], 2) for c in cal_classes})
np.save("work/class_weights_blend3.npy", w)
top = np.argsort(-(T0 * w), 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C1[j] for j in r) for r in top]})
sub.to_csv("work/submission_blend3.csv", index=False)
print("NONE top1 share:", (sub.root_code.str.split().str[0] == "NONE").mean().round(3))
