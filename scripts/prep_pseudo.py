"""Turn the labeller's output on unlabelled rows into a pseudo-labelled training set.

Usage:
    python scripts/prep_pseudo.py <api batch csv> <rows csv> <name>

    api batch csv    output of `api_label transcribe=true` (columns trace_id, codes, transcript, ...)
    rows csv         the rows that were sent (scripts/modal/textdata/<rows>.csv)

All three text columns (gold, ocr_v1, ocr_pub) hold the labeller's transcript. Rows whose transcript is not written
as 'L<n> | ...' lines (blank or irrelevant photos) are dropped.

Writes scripts/modal/textdata/<name>.csv for `train_text extra=<name>`. The rows are used for training only
(fold = -99).
"""

import sys

import pandas as pd

api, rows, name = sys.argv[1:4]

a = pd.read_csv(api)
r = pd.read_csv(rows)
a = a[(a.codes.fillna("") != "") & (a.transcript.fillna("") != "")].copy()
a["root_code"] = a.codes.str.split().str[0]
ok = a.transcript.str.contains(r"^\s*L\d+\s*\|", regex=True)
blank = a[~ok].copy()
a = a[ok]

d = r.merge(a[["trace_id", "root_code", "transcript"]], on="trace_id")
d["gold"] = d.transcript
d["ocr_v1"] = d.transcript
d["ocr_pub"] = d.ocr_v1
d["fold"] = -99

d = d[
    [
        "trace_id",
        "grade",
        "topic_name",
        "question_text",
        "correct_answer",
        "root_code",
        "gold",
        "ocr_v1",
        "ocr_pub",
        "fold",
    ]
]
assert d.trace_id.is_unique
d.to_csv(f"scripts/modal/textdata/{name}.csv", index=False)
print(
    len(d),
    "rows (",
    len(blank),
    "blank-photo transcripts seen ); NONE share",
    round((d.root_code == "NONE").mean(), 3),
    "| codes",
    d.root_code.nunique(),
)
