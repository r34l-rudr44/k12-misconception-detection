"""Modal app `k12-api`: label photos of student work with a Claude model (pseudo-labels for unlabelled rows).

The competition host allowed external APIs for pseudo-labelling the unlabelled set and for data preparation, not at
inference time. Test rows are never sent to the API.

Needs the Modal secret `anthropic` (ANTHROPIC_API_KEY) and the data volume filled by `modal_app.py::fetch_data`.

Usage:
    modal deploy modal_api.py
    export K12_APP=k12-api

    # measure a model against gold labels on 20 fold-0 train rows (synchronous)
    python scripts/modal_spawn.py spawn api_label model=claude-sonnet-5-5 rows=probe_fold0 n=20

    # cheap answer-match screen of unlabelled rows through the Batches API (n=0: all rows)
    python scripts/modal_spawn.py spawn api_label model=claude-haiku-5-5 rows=<rows> task=screen mode=submit n=0 est_per_row=0.00012
    python scripts/modal_spawn.py spawn api_label model=claude-haiku-5-5 rows=<rows> task=screen mode=collect n=0 batch_id=<ids>

    # label + transcript for the selected rows
    python scripts/modal_spawn.py spawn api_label model=claude-sonnet-5-5 rows=<rows> transcribe=true mode=submit n=0 est_per_row=0.0036
    python scripts/modal_spawn.py spawn api_label model=claude-sonnet-5-5 rows=<rows> transcribe=true mode=collect n=0 batch_id=<ids>

`rows` names a CSV in scripts/modal/textdata/ (shipped at deploy time) or in /models/textdata/ on the volume.
Results are written to /models/runs/api/ (`modal volume get k12-models runs/api work/`).

Spend is metered from the API's usage numbers into a ledger on the volume, and a job is refused when the ledger plus
its estimate would pass `cap_total` dollars. Run one job at a time (the ledger is not locked), and collect a batch only
once (collecting adds its cost to the ledger).
"""

import os

import modal

app = modal.App("k12-api")
DATA_VOL = modal.Volume.from_name("k12-data")
MODEL_VOL = modal.Volume.from_name("k12-models")
api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic", "pandas", "pillow")
    .add_local_dir("scripts/modal", remote_path="/root/k12")
)


@app.function(
    image=api_image,
    volumes={"/data": DATA_VOL, "/models": MODEL_VOL},
    secrets=[modal.Secret.from_name("anthropic")],
    timeout=3 * 3600,
    cpu=2,
    memory=4096,
)
def api_label(
    model: str = "claude-haiku-5-5",
    rows: str = "probe_fold0",
    mode: str = "sync",
    n: int = 20,
    cap_total: float = 10.0,
    effort: str = "medium",
    batch_id: str = "",
    est_per_row: float = 0.0,
    transcribe: bool = False,
    task: str = "diagnose",
    max_tokens: int = 6000,
    chunk: int = 750,
    max_side: int = 1568,
):
    """Label train rows (probe) or unlabelled rows. mode = sync | submit | collect; task = diagnose | screen.
    n = number of rows taken from the top of the file (0 = all). cap_total = hard cap on the metered spend in dollars."""
    import sys

    sys.path.insert(0, "/root/k12")
    MODEL_VOL.reload()  # see files uploaded after this container started
    import api_label as al

    on_volume = f"/models/textdata/{rows}.csv"
    res = al.main(
        data="/data/competition_data",
        out_dir="/models/runs/api",
        rows_csv=on_volume if os.path.exists(on_volume) else f"/root/k12/textdata/{rows}.csv",
        model=model,
        mode=mode,
        n=n,
        cap_total=cap_total,
        effort=effort,
        batch_id=batch_id,
        est_per_row=est_per_row,
        transcribe=transcribe,
        task=task,
        max_tokens=max_tokens,
        chunk=chunk,
        max_side=max_side,
    )
    MODEL_VOL.commit()
    return res
