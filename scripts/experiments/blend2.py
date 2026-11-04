"""Early experiment: blend the CPU pipeline (public OCR) with the public baseline's probabilities, with a scale on the
NONE column, tuned on out-of-fold MAP@3. Superseded by scripts/blend_cpu.py.

Usage:
    python scripts/experiments/blend2.py
"""

import json

import numpy as np
import pandas as pd

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
assert list(bm["train_ids"]) == list(tr.trace_id) and list(bm["test_ids"]) == list(te.trace_id)
# map the baseline columns into our class order
B2 = np.zeros_like(A_oof)
B2t = np.zeros_like(A_te)
for j, c in enumerate(C2):
    if c in C1:
        B2[:, C1.index(c)] = B_oof[:, j]
        B2t[:, C1.index(c)] = B_te[:, j]
y = tr.root_code.values
ni = C1.index("NONE")


def map3(P):
    idx = np.argsort(-P, 1)[:, :3]
    s = 0
    for row, t in zip(idx, y):
        p = [C1[j] for j in row]
        s += (1 / (p.index(t) + 1)) if t in p else 0
    return s / len(y)


best = (0, None)
for w in np.linspace(0, 1, 11):
    for ns in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        P = w * A_oof + (1 - w) * B2
        P[:, ni] *= ns
        sc = map3(P)
        if sc > best[0]:
            best = (sc, (w, ns))
print(
    "cpu",
    round(map3(A_oof), 4),
    "baseline",
    round(map3(B2), 4),
    "best blend",
    round(best[0], 4),
    "w_mine,none_scale",
    best[1],
)
w, ns = best[1]
P = w * A_te + (1 - w) * B2t
P[:, ni] *= ns
top = np.argsort(-P, 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C1[j] for j in r) for r in top]})
sub.to_csv("work/submission_blend2.csv", index=False)
print("NONE top1 share:", (sub.root_code.str.split().str[0] == "NONE").mean().round(3))
