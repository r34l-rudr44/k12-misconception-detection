"""Early experiment: alternative ways to combine the VLM diagnosis with the CPU blend (linear, geometric, multiplicative
boost, confident top-1 override, rank filling), compared on the holdout rows. Needs the class weights written by
blend3.py.

Usage:
    python scripts/experiments/blend_vlm2.py <run> [diag prefix = vllm_diag]
"""

import json
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
K = len(C)
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
CPU_te = 0.4 * A_te + 0.6 * B2t
w3 = np.load("work/class_weights_blend3.npy")  # per-class calibration incl. NONE x0.6


def vlm_matrix(path, ids):
    d = pd.read_csv(path).set_index("trace_id")
    M = np.zeros((len(ids), K))
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


V_ho, have = vlm_matrix(f"work/{run}/{prefix}_holdout.csv", tr.trace_id)
idx = np.where(have)[0]
y = np.array([ci[c] for c in tr.root_code])[idx]
V = V_ho[idx]
Cp = CPU_oof[idx] * w3
print(f"n={len(idx)} | VLM {map3(V, y):.4f} | CPU(calibrated) {map3(Cp, y):.4f}")
res = {}
# 1 linear
for w in [0.05, 0.1, 0.2, 0.3]:
    res[f"lin w={w}"] = map3(w * V + (1 - w) * Cp, y)
# 2 log-space (geometric)
for a in [0.1, 0.2, 0.3, 0.5]:
    res[f"geo a={a}"] = map3(np.log(Cp + 1e-6) + a * np.log(V + 0.02), y)
# 3 multiplicative boost
for a in [0.5, 1, 2, 4]:
    res[f"boost a={a}"] = map3(Cp * (1 + a * V), y)
# 4 VLM top-1 override when confident
for t in [0.6, 0.8, 0.95]:
    P = Cp.copy()
    conf = V.max(1) >= t
    P[conf] = 0.5 * P[conf] + 0.5 * V[conf]
    res[f"override t={t}"] = map3(P, y)
# 5 rank-2/3 only from VLM: keep CPU top1, fill next slots with VLM order
P = Cp.copy()
top1 = P.argmax(1)
Pr = V.copy()
Pr[np.arange(len(P)), top1] = 2.0
res["cpu top1 + vlm ranks"] = map3(Pr, y)
for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:8]:
    print(f"{v:.4f}  {k}")
best = max(res, key=res.get)
print("best:", best)
# write test submission for the best boost/geo/linear family
V_te, have_te = vlm_matrix(f"work/{run}/{prefix}_test.csv", te.trace_id)
Ct = CPU_te * w3
kind, val = best.split()[0], float(best.split("=")[-1]) if "=" in best else None
if kind == "lin":
    P = val * V_te + (1 - val) * Ct
elif kind == "geo":
    P = np.exp(np.log(Ct + 1e-6) + val * np.log(V_te + 0.02))
elif kind == "boost":
    P = Ct * (1 + val * V_te)
elif kind == "override":
    P = Ct.copy()
    conf = V_te.max(1) >= val
    P[conf] = 0.5 * P[conf] + 0.5 * V_te[conf]
else:
    P = V_te.copy()
    P[np.arange(len(P)), Ct.argmax(1)] = 2.0
P[~have_te] = Ct[~have_te]
top = np.argsort(-P, 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C[j] for j in r) for r in top]})
sub.to_csv(f"work/submission_{run}_b2.csv", index=False)
print(
    "wrote", f"work/submission_{run}_b2.csv", "NONE top1:", (sub.root_code.str.split().str[0] == "NONE").mean().round(3)
)
