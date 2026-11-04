"""Validate a submission file, then submit it with the Kaggle CLI.

Checks the columns, that the trace ids equal those of the sample submission, and reports codes outside the taxonomy.

Usage:
    python scripts/submit.py work/submission_x.csv "message"
"""

import subprocess
import sys

import pandas as pd

COMP = "misconception-detection-in-k-12-student-math-workings"


def kaggle(*args):
    return subprocess.run([sys.executable, "-m", "kaggle", "competitions", *args], capture_output=True, text=True)


def main(path, msg):
    sub = pd.read_csv(path)
    ss = pd.read_csv("data/sample_submission.csv")
    tax = pd.read_csv("data/taxonomy_v1.csv")
    valid = set(tax.id) | {"NONE"}
    assert list(sub.columns) == ["trace_id", "root_code"], sub.columns
    assert set(sub.trace_id) == set(ss.trace_id) and len(sub) == len(ss), "trace_id mismatch"
    bad = sub.root_code.astype(str).str.split().explode().pipe(lambda s: s[~s.isin(valid)])
    print(
        "rows",
        len(sub),
        "| invalid codes:",
        bad.value_counts().head().to_dict(),
        "| top1 dist:",
        sub.root_code.str.split().str[0].value_counts().head(4).to_dict(),
    )
    r = kaggle("submit", "-c", COMP, "-f", path, "-m", msg)
    print(r.stdout[-500:])
    if r.returncode or "rror" in r.stderr:
        print("STDERR:", r.stderr[-500:])
    assert r.returncode == 0, "kaggle submit failed"
    print(kaggle("submissions", "-c", COMP).stdout[:1200])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
