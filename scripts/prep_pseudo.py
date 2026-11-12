"""Turn the labeller's output on unlabelled rows into a pseudo-labelled training set.

Usage:
    python scripts/prep_pseudo.py <api batch csv> <rows csv> <name> [reader ocr csv]

    api batch csv    output of `api_label transcribe=true` (columns trace_id, codes, transcript, ...)
    rows csv         the rows that were sent (scripts/modal/textdata/<rows>.csv)
    reader ocr csv   optional: our own reader's transcription of the same photos (columns trace_id, ocr), from
                     `vllm_run split='' transcribe=unlabelled ids=<rows>`

With the reader file, pseudo rows get the same kind of noisy text the test rows have (ocr_v1 / ocr_pub = reader output)
and the labeller's clean transcript only in `gold`; rows the reader did not transcribe are dropped. Without it, all
three text columns hold the labeller's transcript, which is cleaner than anything seen at test time.

Blank or irrelevant photos: the labeller writes a sentence instead of 'L<n> | ...' lines. Without reader text these rows
are dropped. With reader text, up to 15 of those labelled REL-01 whose reader text is '[no visible student work ...]'
are kept with the reader text in every column (the test set contains a few such photos).

Writes scripts/modal/textdata/<name>.csv for `train_text extra=<name>` and for `K12_PSEUDO=<name>` in cpu_baseline.py.
The rows are used for training only (fold = -99).
"""

import sys

import pandas as pd

api, rows, name = sys.argv[1:4]
reader = sys.argv[4] if len(sys.argv) > 4 else ""

a = pd.read_csv(api)
r = pd.read_csv(rows)
a = a[(a.codes.fillna("") != "") & (a.transcript.fillna("") != "")].copy()
a["root_code"] = a.codes.str.split().str[0]
ok = a.transcript.str.contains(r"^\s*L\d+\s*\|", regex=True)
blank = a[~ok].copy()
a = a[ok]

d = r.merge(a[["trace_id", "root_code", "transcript"]], on="trace_id")
d["gold"] = d.transcript
if reader:
    o = pd.read_csv(reader).set_index("trace_id").ocr
    n0 = len(d)
    d = d[d.trace_id.isin(o.index)].copy()
    d["ocr_v1"] = d.trace_id.map(o)
    print("reader OCR found for", len(d), "of", n0, "(rest dropped)")
    b = blank[(blank.root_code == "REL-01") & blank.trace_id.isin(o.index)]
    b = b[b.trace_id.map(o).str.contains("no visible student work", case=False)].head(15)
    b = r.merge(b[["trace_id", "root_code"]], on="trace_id")
    b["ocr_v1"] = b.trace_id.map(o)
    b["gold"] = b.ocr_v1
    d = pd.concat([d, b], ignore_index=True)
    print("blank-photo REL-01 rows kept:", len(b))
else:
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
