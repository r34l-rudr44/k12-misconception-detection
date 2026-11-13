"""Modal app `k12-cpu`: the whole CPU chain on a 4-core / 16 GB container (about 10 minutes).

    cpu_baseline.py on both OCR sources (in parallel) -> blend_cpu.py -> none_stack.py

Useful when the local machine does not have the memory for the TF-IDF models. No GPU, no API. Hard timeout 60 min.

Usage:
    modal deploy modal_cpu.py
    K12_APP=k12-cpu python scripts/modal_spawn.py spawn cpu_chain pseudo=<pseudo set or ''> name=<run name>
    modal volume get k12-models runs/<run name> work/_modal/

The outputs (*.npy, cpu_meta_*.json, submission_*.csv, chain.log) land in /models/runs/<run name> on the volume.
The local folders scripts/, data/ (CSV files), data/external/baseline_ocr/ and work/v1ocr/ are copied into the image at
deploy time, so redeploy after changing any of them.
"""

import modal

app = modal.App("k12-cpu")
MODEL_VOL = modal.Volume.from_name("k12-models", create_if_missing=True)
P = "/root/proj"
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("scikit-learn==1.9.1", "numpy==2.5.3", "pandas==3.0.6", "scipy==1.18.1")
    .add_local_dir("scripts", remote_path=f"{P}/scripts", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("data/external/baseline_ocr", remote_path=f"{P}/data/external/baseline_ocr")
    .add_local_dir("work/v1ocr", remote_path=f"{P}/work/v1ocr")
)
for f in ["train.csv", "test.csv", "unlabelled.csv", "taxonomy_v1.csv", "sample_submission.csv"]:
    image = image.add_local_file(f"data/{f}", remote_path=f"{P}/data/{f}")


@app.function(image=image, volumes={"/models": MODEL_VOL}, timeout=3600, cpu=4, memory=16384)
def cpu_chain(pseudo: str = "", name: str = "cpu", pseudo_w: float = 0.5):
    """pseudo = name of a pseudo-labelled set in scripts/modal/textdata ('' = train rows only); pseudo_w = its row weight."""
    import glob
    import os
    import shutil
    import subprocess
    import time

    env = dict(os.environ, PYTHONUTF8="1", K12_PSEUDO_W=str(pseudo_w))
    t0 = time.time()
    if pseudo:
        env["K12_PSEUDO"] = pseudo
    log = open(f"{P}/work/chain.log", "w")

    def run(*args):
        return subprocess.Popen(["python", "-X", "utf8", *args], cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT)

    # the two OCR sources in parallel
    a, b = run("scripts/cpu_baseline.py"), run("scripts/cpu_baseline.py", "work/v1ocr", "v1")
    rc = [a.wait(), b.wait()]
    if not any(rc):
        rc.append(run("scripts/blend_cpu.py").wait())
    if not any(rc):
        rc.append(run("scripts/none_stack.py", "work/cpu_blend_oof.npy", "work/cpu_blend_test.npy", "cpu_blend").wait())
    log.close()

    out = f"/models/runs/{name}"
    os.makedirs(out, exist_ok=True)
    for f in (
        glob.glob(f"{P}/work/*.npy")
        + glob.glob(f"{P}/work/*.json")
        + glob.glob(f"{P}/work/submission_*.csv")
        + [f"{P}/work/chain.log"]
    ):
        shutil.copy(f, out)
    MODEL_VOL.commit()
    return {
        "rc": rc,
        "minutes": round((time.time() - t0) / 60, 1),
        "files": sorted(os.listdir(out)),
        "log_tail": open(f"{P}/work/chain.log").read()[-2500:],
    }
