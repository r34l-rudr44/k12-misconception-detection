"""Blend the CPU pipelines of both OCR sources with the public baseline's probabilities.

Inputs (written by `cpu_baseline.py` for each OCR source, plus the public baseline notebook's outputs):
    work/cpu_{oof,test}_qwen3.5-9b.npy      CPU pipeline on the public zero-shot OCR
    work/cpu_{oof,test}_v1.npy              CPU pipeline on the fine-tuned reader's OCR
    data/external/baseline_ocr/probs_*      out-of-fold and test probabilities of the public baseline

The three weights and a scale on the NONE column are grid-searched on 5-fold out-of-fold MAP@3.

Usage:
    python scripts/blend_cpu.py

Writes work/cpu_blend_{oof,test}.npy and work/submission_cpu_blend.csv.
"""

import itertools
import json

import numpy as np
import pandas as pd

BASELINE = "data/external/baseline_ocr"

tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
C = json.load(open("work/cpu_meta_qwen3.5-9b.json"))["classes"]
ci = {c: i for i, c in enumerate(C)}
ni = ci["NONE"]
assert json.load(open("work/cpu_meta_v1.json"))["classes"] == C
y = np.array([ci[c] for c in tr.root_code])


def load(o, t):
    return np.load(o), np.load(t)


# (out-of-fold, test) probability pairs
A = load("work/cpu_oof_qwen3.5-9b.npy", "work/cpu_test_qwen3.5-9b.npy")
A1 = load("work/cpu_oof_v1.npy", "work/cpu_test_v1.npy")

# the baseline has its own class order: map its columns into ours
bm = json.load(open(f"{BASELINE}/probs_meta_baseline.json"))
Bo, Bt = np.load(f"{BASELINE}/probs_oof_baseline.npy"), np.load(f"{BASELINE}/probs_test_baseline.npy")
B = (np.zeros_like(A[0]), np.zeros_like(A[1]))
for j, c in enumerate(bm["classes"]):
    if c in ci:
        B[0][:, ci[c]] = Bo[:, j]
        B[1][:, ci[c]] = Bt[:, j]


def map3(P, y):
    top = np.argsort(-P, 1)[:, :3]
    return float(np.mean((top == y[:, None]) @ np.array([1, 1 / 2, 1 / 3])))


best = (0, None)
for wa, wa1 in itertools.product(np.arange(0, 1.01, 0.1), repeat=2):
    wb = 1 - wa - wa1
    if wb < -1e-9:
        continue
    P = wa * A[0] + wa1 * A1[0] + wb * B[0]
    for s in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        Q = P.copy()
        Q[:, ni] *= s
        m = map3(Q, y)
        if m > best[0]:
            best = (m, (round(float(wa), 2), round(float(wa1), 2), round(float(wb), 2), s))
print("components OOF:", {k: round(map3(v[0], y), 4) for k, v in dict(A=A, A1=A1, B=B).items()})
print("best blend OOF %.4f  (w_pubOCR, w_v1OCR, w_baseline, NONE scale) = %s" % best)

wa, wa1, wb, s = best[1]
Po = wa * A[0] + wa1 * A1[0] + wb * B[0]
Pt = wa * A[1] + wa1 * A1[1] + wb * B[1]
Po[:, ni] *= s
Pt[:, ni] *= s
np.save("work/cpu_blend_oof.npy", Po)
np.save("work/cpu_blend_test.npy", Pt)
top = np.argsort(-Pt, 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C[j] for j in r) for r in top]})
sub.to_csv("work/submission_cpu_blend.csv", index=False)
print("wrote work/submission_cpu_blend.csv; NONE top1", round((top[:, 0] == ni).mean(), 3))
