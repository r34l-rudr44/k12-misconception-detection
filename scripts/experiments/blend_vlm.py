"""Early experiment: blend the VLM's direct diagnosis (work/<run>/vllm_diag_*.csv or diag_*.csv) with the CPU blend.

The VLM weight and the NONE scale are tuned on the holdout rows, where both VLM and out-of-fold CPU probabilities
exist. Reports MAP@3 for VLM alone, CPU alone and the blend, and writes work/submission_<run>.csv when a test file
exists. The direct diagnosis added little; the final pipeline uses the VLM as a reader only.

Usage:
    python scripts/experiments/blend_vlm.py <run> [diag prefix = vllm_diag]
"""

import json
import os
import sys

import numpy as np
import pandas as pd

run = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "vllm_diag"
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
mm = json.load(open("work/cpu_meta_qwen3.5-9b.json"))
C = mm["classes"]
ci = {c: i for i, c in enumerate(C)}
ni = ci["NONE"]
A_oof = np.load("work/cpu_oof_qwen3.5-9b.npy")
A_te = np.load("work/cpu_test_qwen3.5-9b.npy")
bm = json.load(open("data/external/baseline_ocr/probs_meta_baseline.json"))
C2 = bm["classes"]
B_oof = np.load("data/external/baseline_ocr/probs_oof_baseline.npy")
B_te = np.load("data/external/baseline_ocr/probs_test_baseline.npy")
B2 = np.zeros_like(A_oof)
B2t = np.zeros_like(A_te)
for j, c in enumerate(C2):
    if c in ci:
        B2[:, ci[c]] = B_oof[:, j]
        B2t[:, ci[c]] = B_te[:, j]
CPU_oof = 0.4 * A_oof + 0.6 * B2
CPU_te = 0.4 * A_te + 0.6 * B2t  # = blend2 before NONE scaling


def vlm_matrix(path, ids):
    d = pd.read_csv(path).set_index("trace_id")
    M = np.zeros((len(ids), len(C)))
    for i, t in enumerate(ids):
        if t in d.index:
            for c, p in json.loads(d.loc[t, "probs"]).items():
                if c in ci:
                    M[i, ci[c]] = p
    return M, ids.isin(d.index).values


def map3(P, y):
    top = np.argsort(-P, 1)[:, :3]
    s = 0
    for row, t in zip(top, y):
        s += (1 / (list(row).index(t) + 1)) if t in row else 0
    return s / len(y)


ho_path = f"work/{run}/{prefix}_holdout.csv"
V_ho, have = vlm_matrix(ho_path, tr.trace_id)
idx = np.where(have)[0]
y = np.array([ci[c] for c in tr.root_code])[idx]
V = V_ho[idx]
Cp = CPU_oof[idx]
print(
    f"holdout rows with VLM preds: {len(idx)} | VLM alone {map3(V, y):.4f} | CPU alone {map3(Cp, y):.4f} | CPU none x0.6 {map3(Cp * np.where(np.arange(len(C)) == ni, 0.6, 1), y):.4f}"
)
best = (0, None)
for w in np.linspace(0, 1, 21):
    for ns in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
        P = w * V + (1 - w) * Cp
        P = P * np.where(np.arange(len(C)) == ni, ns, 1)
        sc = map3(P, y)
        if sc > best[0]:
            best = (sc, (round(w, 2), ns))
print(f"best blend on holdout: {best[0]:.4f} (w_vlm, none_scale) = {best[1]}")
te_path = f"work/{run}/{prefix}_test.csv"
if os.path.exists(te_path):
    w, ns = best[1]
    V_te, have_te = vlm_matrix(te_path, te.trace_id)
    P = w * V_te + (1 - w) * CPU_te
    P[~have_te] = CPU_te[~have_te]
    P = P * np.where(np.arange(len(C)) == ni, ns, 1)
    top = np.argsort(-P, 1)[:, :3]
    sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C[j] for j in r) for r in top]})
    sub.to_csv(f"work/submission_{run}.csv", index=False)
    print(
        "wrote",
        f"work/submission_{run}.csv",
        "| test rows with VLM:",
        have_te.sum(),
        "| NONE top1:",
        (sub.root_code.str.split().str[0] == "NONE").mean().round(3),
    )
