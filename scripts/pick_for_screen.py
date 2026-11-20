"""Choose unlabelled rows for the cheap answer-match screen (the first step of a pseudo-labelling round).

Only questions that also occur in the test set are useful. Each question gets a quota proportional to its number of
test rows, with a floor of `min per question` and capped by the rows available; rows are sampled at random inside a
question. Rows that already went through the API (any file in work/api/) are skipped.

Usage:
    python scripts/pick_for_screen.py <n rows> <out name> [min per question = 8] [seed = 42]

Writes scripts/modal/textdata/<out name>.csv, the input of `api_label task=screen`.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

TEXTDATA = "scripts/modal/textdata"

n, name = int(sys.argv[1]), sys.argv[2]
floor = int(sys.argv[3]) if len(sys.argv) > 3 else 8
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

un = pd.read_csv("data/unlabelled.csv")
te = pd.read_csv("data/test.csv")

seen = set()
for f in glob.glob("work/api/api_*.csv"):
    seen |= set(pd.read_csv(f, usecols=["trace_id"]).trace_id)
pool = un[un.question_id.isin(set(te.question_id)) & ~un.trace_id.isin(seen)]

avail = pool.question_id.value_counts()
tq = te.question_id.value_counts().reindex(avail.index).values.astype(float)
av = avail.values


def quota(level):
    return np.minimum(av, np.maximum(floor, np.round(level * tq)))


lo, hi = 0.0, 1000.0
for _ in range(60):  # bisection on the level (rows per test row) that gives n rows in total
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if quota(mid).sum() < n else (lo, mid)
take = dict(zip(avail.index, quota(hi).astype(int)))

rng = np.random.RandomState(seed)
picks = pd.concat([g.sample(take[q], random_state=rng) for q, g in pool.groupby("question_id", sort=True)])
picks = picks.sample(frac=1, random_state=rng).head(n)

cols = ["trace_id", "question_id", "grade", "topic_name", "question_text", "correct_answer", "image_file"]
assert picks.trace_id.is_unique and not picks.trace_id.isin(te.trace_id).any()
os.makedirs(TEXTDATA, exist_ok=True)
picks[cols].to_csv(f"{TEXTDATA}/{name}.csv", index=False)
covered = te.question_id.isin(set(picks.question_id)).sum()
print(
    f"wrote {name}.csv: {len(picks)} rows over {picks.question_id.nunique()} questions |",
    f"level {hi:.2f} rows per test row, floor {floor} | test rows covered {covered} of {len(te)}",
)
