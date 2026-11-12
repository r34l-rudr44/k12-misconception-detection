"""Modal app `k12-misconception`: data download and all GPU jobs.

    fetch_data      CPU   pull the competition data (gated Hugging Face mirror) into the volume `k12-data`
    train_vlm       GPU   LoRA fine-tune Qwen3-VL-4B on two tasks: transcribe the photo, diagnose the root code
    merge_adapter   CPU   merge the LoRA adapter into the base weights (vLLM serves merged models)
    vllm_run        GPU   vLLM inference with the merged model: transcription (the "reader") and direct diagnosis
    infer_vlm       GPU   the same diagnosis with transformers beam search (slower; kept for small holdouts)
    train_text      GPU   Qwen3-8B + LoRA + classification head on the transcribed text, one fold per call
    predict_text    GPU   test-set inference from a saved text-classifier adapter
    pull / download       copy the small result files of a run to work/<run>/

Usage:
    modal deploy modal_app.py
    python scripts/modal_spawn.py spawn train_text run=p2 fold=0 extra=pseudo_mix
    python scripts/modal_spawn.py poll <call id>
    modal run modal_app.py::download --run p2

Needs the Modal secret `huggingface` (HF_TOKEN of an account that accepted the dataset's terms). Outputs go to
/models/runs/<run> on the volume `k12-models`. The folder scripts/modal/ is copied into the images at deploy time, so
redeploy after changing code or scripts/modal/textdata/.

Every function has a hard timeout. GPU containers can be preempted, and Modal then restarts the call
from the beginning; `train_text` checkpoints to the volume and resumes.

Environment variables read at deploy time: K12_GPU (default A10G), K12_BASE_MODEL (the VLM), K12_TRAIN_TIMEOUT and
K12_TEXT_TIMEOUT (seconds).
"""

import os

import modal

APP = "k12-misconception"
DATA_VOL = modal.Volume.from_name("k12-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("k12-models", create_if_missing=True)
HF_DATASET = "colearnlab/misconception-detection"
BASE_MODEL = os.environ.get("K12_BASE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
GPU = os.environ.get("K12_GPU", "A10G")
TIMEOUTS = {"fetch": 3 * 3600, "train": int(os.environ.get("K12_TRAIN_TIMEOUT", 2 * 3600)), "infer": 1 * 3600 + 1800}

app = modal.App(APP)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]==0.35.3", "pandas")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "peft==0.17.1",
        "bitsandbytes==0.48.1",
        "qwen-vl-utils==0.0.14",
        "pillow",
        "pandas",
        "numpy",
        "scikit-learn",
        "huggingface_hub[hf_transfer]==0.35.3",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir("scripts/modal", remote_path="/root/k12")
)

# the HF secret (HF_TOKEN) is only needed by fetch_data; it must exist in the Modal workspace
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=cpu_image, volumes={"/data": DATA_VOL}, secrets=secrets, timeout=TIMEOUTS["fetch"], cpu=4, memory=8192
)
def fetch_data(force: bool = False):
    """Download the dataset snapshot, unzip it and normalise the layout to
    /data/competition_data/{train.csv, test.csv, ..., images/{train,test,questions,unlabelled}}."""
    import glob
    import shutil

    from huggingface_hub import list_repo_files, snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    files = list_repo_files(HF_DATASET, repo_type="dataset")
    print("repo files:", files)
    if (
        os.path.exists("/data/competition_data/train.csv")
        and os.path.isdir("/data/competition_data/images/test")
        and not force
    ):
        n = len(os.listdir("/data/competition_data/images/test"))
        print("already present, test images:", n)
        if n >= 3498:
            return "present"
    zips = [f for f in files if f.endswith(".zip")]
    snapshot_download(HF_DATASET, repo_type="dataset", local_dir="/data/_hf", allow_patterns=["*.csv"] + zips)
    DATA_VOL.commit()
    import zipfile

    for z in zips:
        print("unzipping", z, flush=True)
        with zipfile.ZipFile(f"/data/_hf/{z}") as zf:
            zf.extractall("/data/_unz")
    # normalise layout -> /data/competition_data/{train.csv,...,images/{train,test,questions,unlabelled}}
    cand = glob.glob("/data/_unz/**/train.csv", recursive=True)
    src = os.path.dirname(cand[0]) if cand else "/data/_unz"
    print("unzipped root:", src, os.listdir(src)[:10])
    if os.path.exists("/data/competition_data"):
        shutil.rmtree("/data/competition_data")
    shutil.move(src, "/data/competition_data")
    for c in glob.glob("/data/_hf/*.csv"):
        dst = f"/data/competition_data/{os.path.basename(c)}"
        if not os.path.exists(dst):
            shutil.copy(c, dst)
    for z in zips:
        os.remove(f"/data/_hf/{z}")
    shutil.rmtree("/data/_unz", ignore_errors=True)
    out = {
        d: len(os.listdir(f"/data/competition_data/images/{d}")) for d in os.listdir("/data/competition_data/images")
    }
    print("image counts:", out)
    DATA_VOL.commit()
    return out


@app.function(image=gpu_image, timeout=600)
def smoke():
    """Check that the GPU image imports and that the remote modules load."""
    import peft
    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
    import sys

    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401  (architecture available?)

    sys.path.insert(0, "/root/k12")
    import common  # noqa: F401
    import infer_lora  # noqa: F401
    import train_lora  # noqa: F401

    return "ok"


@app.function(
    image=gpu_image, gpu=GPU, volumes={"/data": DATA_VOL, "/models": MODEL_VOL}, timeout=TIMEOUTS["train"], memory=32768
)
def train_vlm(
    run: str = "v1",
    epochs: float = 2.0,
    limit: int = 0,
    max_pixels: int = 768 * 768,
    lr: float = 1e-4,
    holdout_frac: float = 0.2,
    batch: int = 2,
    accum: int = 4,
    lora_r: int = 32,
    tax_in_prompt: bool = True,
    tasks: str = "diagnose,transcribe",
    none_keep: float = 1.0,
):
    """LoRA fine-tune of the VLM on `tasks`. 20% of train is held out (stratified). none_keep < 1 down-samples NONE rows
    in the diagnosis task, because the model otherwise predicts NONE far too often."""
    import sys

    sys.path.insert(0, "/root/k12")
    import train_lora

    res = train_lora.main(
        base=BASE_MODEL,
        data="/data/competition_data",
        out=f"/models/runs/{run}",
        epochs=epochs,
        limit=limit,
        max_pixels=max_pixels,
        lr=lr,
        holdout_frac=holdout_frac,
        batch=batch,
        accum=accum,
        lora_r=lora_r,
        tax_in_prompt=tax_in_prompt,
        tasks=tasks.split(","),
        none_keep=none_keep,
    )
    MODEL_VOL.commit()
    return res


@app.function(
    image=gpu_image, gpu=GPU, volumes={"/data": DATA_VOL, "/models": MODEL_VOL}, timeout=TIMEOUTS["infer"], memory=32768
)
def infer_vlm(
    run: str = "v1",
    split: str = "holdout,test",
    limit: int = 0,
    max_pixels: int = 768 * 768,
    beams: int = 8,
    transcribe_n: int = 100,
    batch: int = 4,
):
    """Diagnosis with transformers beam search over the root code, plus transcription of `transcribe_n` holdout rows."""
    import sys

    sys.path.insert(0, "/root/k12")
    import infer_lora

    res = infer_lora.main(
        base=BASE_MODEL,
        data="/data/competition_data",
        run_dir=f"/models/runs/{run}",
        splits=split.split(","),
        limit=limit,
        max_pixels=max_pixels,
        beams=beams,
        transcribe_n=transcribe_n,
        batch=batch,
    )
    MODEL_VOL.commit()
    return res


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "pandas", "numpy", "pillow", "scikit-learn", "huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_dir("scripts/modal", remote_path="/root/k12")
)


@app.function(image=vllm_image, timeout=600)
def vllm_smoke():
    """Check that the vLLM image imports."""
    import torch
    import transformers
    import vllm

    print("vllm", vllm.__version__, "transformers", transformers.__version__, "torch", torch.__version__)
    return "ok"


@app.function(image=gpu_image, volumes={"/models": MODEL_VOL}, timeout=1800, cpu=8, memory=40000)
def merge_adapter(run: str = "v1"):
    """CPU: merge the LoRA adapter into the base weights -> /models/runs/<run>/merged (for vLLM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    out = f"/models/runs/{run}/merged"
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, f"/models/runs/{run}/adapter").merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(out)
    MODEL_VOL.commit()
    return os.listdir(out)


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={"/data": DATA_VOL, "/models": MODEL_VOL},
    timeout=TIMEOUTS["infer"],
    memory=32768,
)
def vllm_run(
    run: str = "v1",
    split: str = "holdout,test",
    limit: int = 0,
    max_pixels: int = 768 * 768,
    n_samples: int = 12,
    transcribe: str = "holdout",
    transcribe_limit: int = 0,
    tax_in_prompt: bool = False,
    use_base: bool = False,
    beams: int = 0,
    ids: str = "",
):
    """vLLM inference with the merged model of `run` (or the base model with use_base).

    split        splits to diagnose, comma separated: holdout, train, test ('' = none)
    n_samples    diagnosis = greedy answer + this many samples at T=1, turned into code frequencies (0 = greedy only)
    beams        > 0: beam search instead of sampling (much slower in vLLM)
    transcribe   splits to transcribe: holdout, train, test, unlabelled
    ids          name of a csv in scripts/modal/textdata (column trace_id): with transcribe='unlabelled' only those rows
                 are read, e.g. split='' transcribe='unlabelled' ids='pseudo_mix'
                 -> /models/runs/<run>/vllm_ocr_unlabelled_pseudo_mix.csv
    """
    import sys

    sys.path.insert(0, "/root/k12")
    import vllm_infer

    model_dir = BASE_MODEL if use_base else f"/models/runs/{run}/merged"
    res = vllm_infer.main(
        model_dir=model_dir,
        data="/data/competition_data",
        run_dir=f"/models/runs/{run}",
        splits=split.split(","),
        limit=limit,
        max_pixels=max_pixels,
        n_samples=n_samples,
        transcribe_splits=[s for s in transcribe.split(",") if s],
        transcribe_limit=transcribe_limit,
        tax_in_prompt=tax_in_prompt,
        beams=beams,
        ids_csv=f"/root/k12/textdata/{ids}.csv" if ids else "",
    )
    MODEL_VOL.commit()
    return res


@app.function(image=cpu_image, volumes={"/models": MODEL_VOL}, timeout=600)
def pull(run: str = "v1"):
    """Return the small prediction files of a run so they can be written locally."""
    import glob

    out = {}
    for p in glob.glob(f"/models/runs/{run}/*.csv") + glob.glob(f"/models/runs/{run}/*.json"):
        out[os.path.basename(p)] = open(p, encoding="utf-8").read()
    return out


@app.local_entrypoint()
def download(run: str = "v1"):
    """Local entry point: write the csv and json files of a run to work/<run>/."""
    files = pull.remote(run)
    os.makedirs(f"work/{run}", exist_ok=True)
    for k, v in files.items():
        open(f"work/{run}/{k}", "w", encoding="utf-8").write(v)
        print("wrote", f"work/{run}/{k}", len(v))


@app.function(
    image=gpu_image,
    gpu=GPU,
    volumes={"/models": MODEL_VOL},
    timeout=int(os.environ.get("K12_TEXT_TIMEOUT", 3600)),
    memory=32768,
)
def train_text(
    run: str = "t1",
    fold: int = 0,
    base: str = "Qwen/Qwen3-8B",
    epochs: float = 2.0,
    lr: float = 2e-4,
    lora_r: int = 16,
    batch: int = 4,
    accum: int = 2,
    max_len: int = 640,
    p_gold: float = 0.25,
    p_pub: float = 0.25,
    limit: int = 0,
    extra: str = "",
    skip_test: bool = False,
    seed: int = 42,
    tag: str = "",
):
    """Text-only Qwen3 + LoRA + classification head, one fold per call (fold = -1 trains on all rows).

    extra = name of a pseudo-labelled set in scripts/modal/textdata, added to the training rows.
    p_gold / p_pub = probability that a training row shows the human transcription / the public OCR instead of our
    reader's OCR, so the model sees the same kind of noisy text it gets at test time.
    """
    import sys

    sys.path.insert(0, "/root/k12")
    import text_cls

    try:  # the adapter and every prediction file are committed as they are written, so a late failure keeps them
        return text_cls.main(
            base=base,
            data_dir="/root/k12/textdata",
            out=f"/models/runs/{run}",
            fold=fold,
            epochs=epochs,
            lr=lr,
            lora_r=lora_r,
            batch=batch,
            accum=accum,
            max_len=max_len,
            p_gold=p_gold,
            p_pub=p_pub,
            limit=limit,
            extra=extra,
            skip_test=skip_test,
            seed=seed,
            tag=tag,
            on_save=MODEL_VOL.commit,
        )
    finally:
        MODEL_VOL.commit()


@app.function(image=gpu_image, gpu=GPU, volumes={"/models": MODEL_VOL}, timeout=2400, memory=32768)
def predict_text(
    run: str = "p2", adapter_tag: str = "fold0", base: str = "Qwen/Qwen3-8B", check_fold: int = 0, max_len: int = 1024
):
    """Test-set inference only, from a saved adapter (/models/runs/<run>/<adapter_tag>_adapter).
    check_fold = the adapter's validation fold (its MAP@3 is re-measured first), or -1 for a full-data adapter
    (adapter_tag is then 'fold-1'). Output files: <adapter_tag>_pred_test_{ocr_v1,ocr_pub}.csv."""
    import sys

    sys.path.insert(0, "/root/k12")
    import text_cls

    MODEL_VOL.reload()
    try:
        return text_cls.predict_only(
            base=base,
            data_dir="/root/k12/textdata",
            out=f"/models/runs/{run}",
            adapter=f"/models/runs/{run}/{adapter_tag}_adapter",
            tag=f"{adapter_tag}_pred",
            max_len=max_len,
            check_fold=check_fold,
            on_save=MODEL_VOL.commit,
        )
    finally:
        MODEL_VOL.commit()
