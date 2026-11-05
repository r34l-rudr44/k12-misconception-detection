"""Bundle the text inputs of the Qwen3-8B classifier.

Each row carries the question, the answer key and up to three versions of the student's working:
    gold      human transcription (train only)
    ocr_v1    our fine-tuned reader (work/v1ocr/, see scripts/run_reader_chain.sh)
    ocr_pub   public zero-shot OCR from the baseline notebook (data/external/baseline_ocr/)

The 5 folds are stratified on the root code; codes with fewer than 5 rows are grouped as RARE.

Usage:
    python scripts/prep_textdata.py

Writes into scripts/modal/textdata/ (this folder is shipped to the Modal containers with the code):
    text_train.csv, text_test.csv, classes.json
    probe_fold0.csv    the fold-0 train rows (shuffled) in the format the labeller reads, to measure it against gold labels
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

OUT = "scripts/modal/textdata"
BASELINE = "data/external/baseline_ocr"

tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
C = json.load(open("work/cpu_meta_qwen3.5-9b.json"))["classes"]


def ocr(path):
    return pd.read_csv(path).set_index("trace_id").ocr


v1t, v1s = ocr("work/v1ocr/ocr_train_v1.csv"), ocr("work/v1ocr/ocr_test_v1.csv")
pbt, pbs = ocr(f"{BASELINE}/ocr_train_qwen3.5-9b.csv"), ocr(f"{BASELINE}/ocr_test_qwen3.5-9b.csv")

cols = ["trace_id", "grade", "topic_name", "question_text", "correct_answer"]
a = tr[cols + ["root_code"]].copy()
a["gold"] = tr.transcription_steps
a["ocr_v1"] = a.trace_id.map(v1t)
a["ocr_pub"] = a.trace_id.map(pbt)
vc = tr.root_code.value_counts()
strat = np.where(tr.root_code.map(vc) >= 5, tr.root_code, "RARE")
a["fold"] = -1
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(a, strat)):
    a.loc[v, "fold"] = k

b = te[cols].copy()
b["ocr_v1"] = b.trace_id.map(v1s)
b["ocr_pub"] = b.trace_id.map(pbs)

assert set(a.root_code) <= set(C)
os.makedirs(OUT, exist_ok=True)
a.to_csv(f"{OUT}/text_train.csv", index=False)
b.to_csv(f"{OUT}/text_test.csv", index=False)
json.dump(C, open(f"{OUT}/classes.json", "w"))

probe_cols = ["trace_id", "question_id", "grade", "topic_name", "question_text", "correct_answer", "image_file"]
tr.loc[(a.fold == 0).values, probe_cols].sample(frac=1, random_state=0).to_csv(f"{OUT}/probe_fold0.csv", index=False)

print(
    len(C),
    "classes | folds",
    a.fold.value_counts().sort_index().tolist(),
    "| missing ocr_v1 train/test",
    a.ocr_v1.isna().sum(),
    b.ocr_v1.isna().sum(),
    "| missing pub",
    a.ocr_pub.isna().sum(),
    b.ocr_pub.isna().sum(),
)
