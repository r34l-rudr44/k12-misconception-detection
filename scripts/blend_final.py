"""Final blend: Qwen3-8B text classifier x CPU blend.

The 8B test probabilities are the average over the two OCR sources (fine-tuned reader and public OCR). The blend is
linear: w * P_8B + (1 - w) * P_CPU. w = 0.5 was chosen on fold 0 and confirmed on the leaderboard.

Usage:
    python scripts/blend_final.py <run> <tag> [w=0.5] [cpu test npy] [name suffix]
    python scripts/blend_final.py p2 fold0_pred

    run   directory under work/ holding the 8B prediction files (downloaded from Modal)
    tag   file prefix of the 8B test files: `fold0_pred` (predict_text on the fold-0 adapter) or `fold-1` (full-data run)

Writes work/submission_<run>_<tag>_w<w><suffix>.csv.
"""

import json
import sys

import numpy as np
import pandas as pd

run, tag = sys.argv[1], sys.argv[2]
w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
cpu = sys.argv[4] if len(sys.argv) > 4 else "work/cpu_blend_ns_test.npy"
suf = sys.argv[5] if len(sys.argv) > 5 else ""

te = pd.read_csv("data/test.csv")
C = json.load(open("work/cpu_meta_qwen3.5-9b.json"))["classes"]
ni = C.index("NONE")
P8 = (
    sum(
        pd.read_csv(f"work/{run}/{tag}_test_{c}.csv").set_index("trace_id").loc[te.trace_id, C].values
        for c in ["ocr_v1", "ocr_pub"]
    )
    / 2
)
Pc = np.load(cpu)
Pc = Pc / Pc.sum(1, keepdims=True)

# Guards: a broken 8B model gives near-uniform probabilities and the blend would silently fall back to the CPU model.
conf, agree, none8 = P8.max(1).mean(), (P8.argmax(1) == Pc.argmax(1)).mean(), (P8.argmax(1) == ni).mean()
print(f"8B: mean max-prob {conf:.3f} | NONE top1 {none8:.3f} | top-1 agreement with CPU blend {agree:.3f}")
assert abs(P8.sum(1) - 1).max() < 0.01 and conf > 0.3 and 0.15 < none8 < 0.6 and agree > 0.4, (
    "8B test probabilities look broken"
)

P = w * P8 + (1 - w) * Pc
top = np.argsort(-P, 1)[:, :3]
sub = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(C[j] for j in r) for r in top]})
out = f"work/submission_{run}_{tag}_w{w}{suf}.csv"
sub.to_csv(out, index=False)
print(
    "wrote",
    out,
    "| rows",
    len(sub),
    "| NONE top1",
    round((sub.root_code.str.split().str[0] == "NONE").mean(), 3),
    "| top-1 changed vs CPU-only",
    round(float((top[:, 0] != Pc.argmax(1)).mean()), 3),
)
