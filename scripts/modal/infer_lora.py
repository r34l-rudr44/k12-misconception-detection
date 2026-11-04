"""Inference with the LoRA adapter through transformers: beam search over the root code (diagnose) and transcription of
a holdout sample. Slower than the vLLM path (vllm_infer.py); kept for small holdout checks.
"""

import difflib
import json
import re
import time

import numpy as np
import pandas as pd
import torch

from common import (
    DIAG_SYS,
    TRANS_SYS,
    build_messages,
    diag_user_text,
    img_path,
    load_image,
    load_tables,
    taxonomy_block,
    trans_user_text,
)


def parse_code(s, valid):
    m = re.search(r"ROOT:\s*([A-Za-z]{2,4}-\d{2}|NONE)", s)
    if m:
        c = m.group(1).upper()
        c = "NONE" if c == "NONE" else c
        for v in valid:
            if v.upper() == c:
                return v
    return None


def main(
    base, data, run_dir, splits=("holdout", "test"), limit=0, max_pixels=768 * 768, beams=8, transcribe_n=100, batch=4
):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    tr, te, tax = load_tables(data)
    valid = sorted(set(tax.id) | {"NONE"})
    tax_block = taxonomy_block(tax) if json.load(open(f"{run_dir}/train_meta.json")).get("tax_in_prompt", True) else ""
    processor = AutoProcessor.from_pretrained(base)
    ip = processor.image_processor
    ip.size = {"shortest_edge": 256 * 32 * 32, "longest_edge": max_pixels}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, f"{run_dir}/adapter")
    model = model.merge_and_unload()
    model.eval()

    def run_split(name):
        if name == "holdout":
            ho = pd.read_csv(f"{run_dir}/holdout.csv")
            df = tr[tr.trace_id.isin(ho.trace_id)].reset_index(drop=True)
            split = "train"
        else:
            df = te.copy()
            split = "test"
        have = df.image_file.map(lambda f: img_path(data, split, f) is not None)
        if have.mean() < 1:
            print(f"WARNING {name}: only {have.sum()}/{len(df)} images present", flush=True)
            df = df[have].reset_index(drop=True)
        if limit:
            df = df.head(limit)
        rows, t0 = [], time.time()
        for b in range(0, len(df), batch):
            chunk = df.iloc[b : b + batch]
            imgs = [load_image(img_path(data, split, f), max_pixels) for f in chunk.image_file]
            texts = [
                processor.apply_chat_template(
                    build_messages(DIAG_SYS, diag_user_text(r, tax_block), im),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for (_, r), im in zip(chunk.iterrows(), imgs)
            ]
            enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                try:
                    gen = model.generate(
                        **enc,
                        max_new_tokens=8,
                        num_beams=beams,
                        num_return_sequences=beams,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                        length_penalty=0.0,
                    )
                    seqs = gen.sequences[:, enc["input_ids"].shape[1] :]
                    scores = gen.sequences_scores.float().cpu().numpy()
                    dec = processor.batch_decode(seqs, skip_special_tokens=True)
                    nb = beams
                except Exception as e:
                    print("beam search failed, falling back to greedy:", repr(e)[:200], flush=True)
                    torch.cuda.empty_cache()
                    gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
                    dec = processor.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
                    scores = np.zeros(len(dec))
                    nb = 1
            for i, (_, r) in enumerate(chunk.iterrows()):
                cand = {}
                for j in range(nb):
                    c = parse_code(dec[i * nb + j], valid)
                    if c:
                        cand[c] = cand.get(c, 0.0) + float(np.exp(scores[i * nb + j]))
                tot = sum(cand.values()) or 1.0
                cand = {k: v / tot for k, v in sorted(cand.items(), key=lambda kv: -kv[1])}
                rows.append(
                    {
                        "trace_id": r.trace_id,
                        "pred": " ".join(list(cand)[:3]),
                        "probs": json.dumps(cand),
                        "raw": dec[i * nb][:60],
                    }
                )
            if (b // batch) % 25 == 0:
                el = time.time() - t0
                print(f"{name} {b + len(chunk)}/{len(df)} {el / (b + len(chunk)):.2f}s/img", flush=True)
        res = pd.DataFrame(rows)
        res.to_csv(f"{run_dir}/diag_{name}.csv", index=False)
        if name == "holdout":
            m = res.merge(tr[["trace_id", "root_code"]], on="trace_id")
            s = 0.0
            for p, t in zip(m.pred.str.split(), m.root_code):
                p = [x for x in (p if isinstance(p, list) else [])]
                s += (1 / (p.index(t) + 1)) if t in p else 0
            print(
                f"HOLDOUT MAP@3 = {s / len(m):.4f} on {len(m)} rows | top1 acc {(m.pred.str.split().str[0] == m.root_code).mean():.3f}",
                flush=True,
            )
            json.dump({"holdout_map3": s / len(m), "n": len(m)}, open(f"{run_dir}/holdout_score.json", "w"))
        return len(res)

    out = {s: run_split(s) for s in splits}

    if transcribe_n:
        ho = pd.read_csv(f"{run_dir}/holdout.csv")
        df = tr[tr.trace_id.isin(ho.trace_id)].head(transcribe_n).reset_index(drop=True)
        rows, sims, t0 = [], [], time.time()
        for b in range(0, len(df), batch):
            chunk = df.iloc[b : b + batch]
            imgs = [load_image(img_path(data, "train", f), max_pixels) for f in chunk.image_file]
            texts = [
                processor.apply_chat_template(
                    build_messages(TRANS_SYS, trans_user_text(r), im), tokenize=False, add_generation_prompt=True
                )
                for (_, r), im in zip(chunk.iterrows(), imgs)
            ]
            enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=384, do_sample=False, repetition_penalty=1.05)
            dec = processor.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for (_, r), d in zip(chunk.iterrows(), dec):
                sims.append(difflib.SequenceMatcher(None, d, str(r.transcription_steps)).ratio())
                rows.append({"trace_id": r.trace_id, "ocr": d})
        pd.DataFrame(rows).to_csv(f"{run_dir}/ocr_holdout.csv", index=False)
        print(
            f"TRANSCRIPTION similarity vs gold on {len(sims)} holdout rows: {np.mean(sims):.3f} ({(time.time() - t0) / len(sims):.2f}s/img)",
            flush=True,
        )
        out["ocr_sim"] = float(np.mean(sims))
    return out
