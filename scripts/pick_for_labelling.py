"""Choose which screened unlabelled rows are sent to the strong model for a label and a transcript.

Input: one or more answer-match screens (cheap model, `api_label task=screen`): each row has the final answer the model
read from the photo and MATCH = YES/NO against the answer key. Only MATCH = NO rows (probably a misconception) are
candidates, because the unlabelled pool is about three quarters NONE and NONE rows add little.

Selection:
  * Per-question quota by water-filling: misconception rows already held for the question (train + earlier pseudo sets)
    plus the new rows should be proportional to the question's number of test rows.
  * Inside a question, rows are taken round-robin over distinct wrong final answers (most common first), so every wrong
    answer gets a label before any gets two.

Usage:
    python scripts/pick_for_labelling.py <n rows> <out name> <screen csv> [more screen csvs ...]

Writes scripts/modal/textdata/<out name>.csv.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer_match as am  # noqa: E402

TEXTDATA = "scripts/modal/textdata"

n, name, screens = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
un = pd.read_csv("data/unlabelled.csv").set_index("trace_id")
te = pd.read_csv("data/test.csv")
tr = pd.read_csv("data/train.csv")
s = pd.concat([pd.read_csv(f) for f in screens], ignore_index=True).drop_duplicates("trace_id")

# rows already sent for labelling (every API result file that is not a screen)
used = set()
for f in glob.glob("work/api/api_*.csv"):
    if "_screen" not in os.path.basename(f):
        used |= set(pd.read_csv(f).trace_id)

s = s[(s.match == "NO") & ~s.trace_id.isin(used) & s.trace_id.isin(un.index)].copy()
s["question_id"] = s.trace_id.map(un.question_id)
s["key"] = s.trace_id.map(un.correct_answer)
s["final"] = s["tail"].fillna("").str.extract(r"FINAL:\s*(.*?)\s*\|\s*MATCH", expand=False).fillna("")

# The screen sometimes says NO although the answer it read equals the key under our own matcher: probably NONE rows.
same = np.array([bool(f and am.match("L1 | " + f, str(k))) for f, k in zip(s.final, s.key)], dtype=bool)
print("flagged rows", len(s), "| dropped (our matcher says the read answer equals the key):", int(same.sum()))
s = s[~same]
s = s[s.question_id.isin(set(te.question_id))]
s["ans"] = s.final.str.lower().str.replace(r"[^0-9a-z,./\-]", "", regex=True).str[:24]

# misconception rows already held per question: train, plus pseudo-labelled sets built in earlier rounds
have = tr[tr.root_code.fillna("NONE") != "NONE"].groupby("question_id").size()
held = [pd.read_csv(f) for f in glob.glob(f"{TEXTDATA}/pseudo*.csv")]
held = [p[["trace_id", "root_code"]] for p in held if "root_code" in p.columns]
if held:
    p = pd.concat(held, ignore_index=True).drop_duplicates("trace_id")
    p = p[p.root_code != "NONE"]
    have = have.add(p.trace_id.map(un.question_id).value_counts(), fill_value=0)

t = te.question_id.value_counts()
avail = s.question_id.value_counts()
qs = avail.index
h = have.reindex(qs).fillna(0).values
tq = t.reindex(qs).values.astype(float)
av = avail.values

lo, hi = 0.0, 1000.0
for _ in range(60):  # bisection on the water level (rows per test row)
    mid = (lo + hi) / 2
    take = np.minimum(av, np.maximum(0, np.round(mid * tq - h)))
    lo, hi = (mid, hi) if take.sum() < n else (lo, mid)
take = np.minimum(av, np.maximum(0, np.round(hi * tq - h))).astype(int)

picks = []
for q, k in zip(qs, take):
    if k == 0:
        continue
    g = s[s.question_id == q]
    order = g.ans.value_counts().index
    buckets = [list(g[g.ans == a].trace_id) for a in order]
    out = []
    while len(out) < k:
        for b in buckets:
            if b and len(out) < k:
                out.append(b.pop(0))
    picks += out
picks = picks[:n]

cols = ["trace_id", "question_id", "grade", "topic_name", "question_text", "correct_answer", "image_file"]
d = un.loc[picks].reset_index()[cols]
assert d.trace_id.is_unique and not d.trace_id.isin(te.trace_id).any() and not d.trace_id.isin(used).any()
os.makedirs(TEXTDATA, exist_ok=True)
d.to_csv(f"{TEXTDATA}/{name}.csv", index=False)
g = s[s.trace_id.isin(picks)]
print(
    f"wrote {name}.csv: {len(d)} rows over {d.question_id.nunique()} questions | water level {hi:.2f} rows per test row |",
    f"distinct (question, wrong answer) pairs {g.groupby(['question_id', 'ans']).ngroups} |",
    f"test rows covered {t.reindex(d.question_id.unique()).sum()} of {len(te)}",
)
