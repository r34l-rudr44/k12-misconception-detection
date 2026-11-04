"""Prompts, image loading and data helpers shared by the Qwen3-VL jobs (train_lora, infer_lora, vllm_infer)."""

import json
import math
import os

import pandas as pd
from PIL import Image, ImageOps

SEED = 42

DIAG_SYS = (
    "You are a maths teacher diagnosing an Indonesian K-12 student's handwritten working from a photo. "
    "Decide the ROOT misconception: the single error the rest of the working follows from, or NONE when the working "
    "is fully correct or has only minor slips that do not change the result. Answer with the taxonomy code first."
)

TRANS_SYS = (
    "You transcribe photos of Indonesian K-12 students' handwritten maths working, verbatim, line by line, "
    "keeping every mistake exactly as written."
)


def load_tables(data):
    tr = pd.read_csv(f"{data}/train.csv")
    te = pd.read_csv(f"{data}/test.csv")
    tax = pd.read_csv(f"{data}/taxonomy_v1.csv")
    return tr, te, tax


def taxonomy_block(tax):
    lines = [f"{r.id}: {r['name']}" for _, r in tax.iterrows()]
    return (
        "Taxonomy codes:\n" + "\n".join(lines) + "\nNONE: no major misconception (correct working or minor slips only)"
    )


def img_path(data, split, fname):
    p = f"{data}/images/{split}/{fname}"
    return p if os.path.exists(p) else None


def load_image(path, max_pixels):
    if not path:
        return Image.new("RGB", (64, 64), "white")
    im = Image.open(path)
    s = math.sqrt(max_pixels / (im.width * im.height))
    if s < 1:
        im.draft("RGB", (int(im.width * s), int(im.height * s)))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = math.sqrt(max_pixels / (im.width * im.height))
    if s < 1:
        im = im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))), Image.BICUBIC)
    return im


def diag_user_text(r, tax_block):
    q = str(r.question_text)[:700]
    parts = [f"Grade {r.grade}. Topic: {r.topic_name}.", f"Question: {q}", f"Answer key: {r.correct_answer}"]
    if tax_block:
        parts.append(tax_block)
    parts.append(
        "Look at the photo of the student's working. Output the root code on the first line as 'ROOT: <code>', "
        "then one line 'WHY: <short explanation>'."
    )
    return "\n".join(parts)


def diag_target(r):
    why = ""
    try:
        for f in json.loads(r.misconceptions) if isinstance(r.misconceptions, str) else []:
            if f.get("root"):
                why = f"{f.get('step', '')} {f.get('span', '')}: {f.get('text', '')}".strip()
                break
    except Exception:
        pass
    if r.root_code == "NONE" and not why:
        why = "The working is correct; no major misconception."
    return f"ROOT: {r.root_code}\nWHY: {why[:400]}"


def trans_user_text(r):
    return (
        f"The question was: {str(r.question_text)[:500]}\n"
        "Transcribe ONLY what the student wrote, line by line, top to bottom, as\nL1 | ...\nL2 | ...\n"
        "Keep mistakes as written, use plain text or simple LaTeX for maths, write [crossed out: ...] for crossed-out text "
        "and [unclear] for illegible parts. Output only the transcription."
    )


def build_messages(sys_text, user_text, image, answer=None):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def holdout_split(tr, frac, seed=SEED):
    """Stratified holdout: rare codes (<5 rows) stay in train."""
    from sklearn.model_selection import train_test_split

    vc = tr.root_code.value_counts()
    strat = tr.root_code.where(tr.root_code.map(vc) >= 5, "RARE")
    idx_tr, idx_ho = train_test_split(tr.index.values, test_size=frac, random_state=seed, stratify=strat)
    return tr.loc[idx_tr].reset_index(drop=True), tr.loc[idx_ho].reset_index(drop=True)
