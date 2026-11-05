"""Re-estimate P(NONE) with a small stacker, then re-rank.

NONE ("no major misconception") is a third of the rows and is almost the same thing as "the student's final answer
equals the answer key". A logistic regression on answer-match features (both OCR sources, last 1/2/3 lines) plus the
blend's own NONE log-odds and confidence predicts P(NONE); the other classes keep their relative order and share the
remaining mass. The stacker is fitted 5-fold out-of-fold, and a final scale on P(NONE) is chosen on out-of-fold MAP@3.

Usage:
    python scripts/none_stack.py <oof.npy> <test.npy> <tag>
    python scripts/none_stack.py work/cpu_blend_oof.npy work/cpu_blend_test.npy cpu_blend

Writes work/<tag>_ns_{oof,test}.npy and work/submission_<tag>_ns.csv.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer_match as am  # noqa: E402

oof_p, te_p, tag = sys.argv[1], sys.argv[2], sys.argv[3]
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
C = json.load(open("work/cpu_meta_qwen3.5-9b.json"))["classes"]
ci = {c: i for i, c in enumerate(C)}
ni = ci["NONE"]
y = np.array([ci[c] for c in tr.root_code])
isn = (y == ni).astype(int)
Po, Pt = np.load(oof_p), np.load(te_p)


def feats(df, P, split):
    """Answer-match flags for both OCR sources and three tail lengths, NONE log-odds and top probability of the blend."""
    cols = []
    for path in [f"work/v1ocr/ocr_{split}_v1.csv", f"data/external/baseline_ocr/ocr_{split}_qwen3.5-9b.csv"]:
        o = df.trace_id.map(pd.read_csv(path).set_index("trace_id").ocr)
        for t in (1, 2, 3):
            cols.append([am.match(x, k, t) for x, k in zip(o, df.correct_answer)])
    F = np.array(cols, float).T
    pn = P[:, ni] / P.sum(1)
    return np.column_stack([F, np.log(pn + 1e-4) - np.log(1 - pn + 1e-4), P.max(1) / P.sum(1)])


def map3(P):
    top = np.argsort(-P, 1)[:, :3]
    return float(np.mean((top == y[:, None]) @ np.array([1, 1 / 2, 1 / 3])))


def apply(P, pn, s):
    """Replace the NONE column by s * pn and rescale the other classes to share 1 - pn."""
    Q = P / P.sum(1, keepdims=True)
    rest = 1 - Q[:, ni]
    Q = Q * ((1 - pn) / np.maximum(rest, 1e-6))[:, None]
    Q[:, ni] = pn * s
    return Q


Fo, Ft = feats(tr, Po, "train"), feats(te, Pt, "test")

pn_oof = np.zeros(len(tr))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Fo, isn):
    pn_oof[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Fo[a], isn[a]).predict_proba(Fo[b])[:, 1]
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Fo, isn)
pn_te = lr.predict_proba(Ft)[:, 1]

print(f"NONE AUC: blend-only {roc_auc_score(isn, Fo[:, -2]):.4f} -> stacked {roc_auc_score(isn, pn_oof):.4f}")
print(f"MAP@3 before {map3(Po):.4f}")
res = {s: map3(apply(Po, pn_oof, s)) for s in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]}
for s, m in res.items():
    print(f"  scale {s}: {m:.4f}")
s = max(res, key=res.get)

Qo, Qt = apply(Po, pn_oof, s), apply(Pt, pn_te, s)
np.save(f"work/{tag}_ns_oof.npy", Qo)
np.save(f"work/{tag}_ns_test.npy", Qt)
top = np.argsort(-Qt, 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C[j] for j in r) for r in top]})
sub.to_csv(f"work/submission_{tag}_ns.csv", index=False)
print(
    f"best scale {s}: OOF {res[s]:.4f}; test NONE top1 {(top[:, 0] == ni).mean():.3f} "
    f"(train NONE share {isn.mean():.3f}); wrote work/submission_{tag}_ns.csv"
)
