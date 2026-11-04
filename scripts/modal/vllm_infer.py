"""vLLM inference with a merged (base + LoRA) Qwen3-VL model.

diagnose     greedy answer plus N samples at T=1 -> code frequencies -> ranking of up to three codes;
             optionally beam search instead of sampling
transcribe   greedy, up to 384 tokens: the reader output used as OCR text by the text models
"""

import difflib
import json
import os
import re
import time

import numpy as np
import pandas as pd

from common import (
    DIAG_SYS,
    SEED,
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
    if not m:
        return None
    c = m.group(1).upper()
    for v in valid:
        if v.upper() == c:
            return v
    return None


def main(
    model_dir,
    data,
    run_dir,
    splits=("holdout", "test"),
    limit=0,
    max_pixels=768 * 768,
    n_samples=12,
    transcribe_splits=("holdout",),
    transcribe_limit=0,
    tax_in_prompt=False,
    gpu_mem=0.9,
    max_model_len=3072,
    beams=0,
    ids_csv="",
):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    tr, te, tax = load_tables(data)
    valid = sorted(set(tax.id) | {"NONE"})
    tax_block = taxonomy_block(tax) if tax_in_prompt else ""
    processor = AutoProcessor.from_pretrained(model_dir)
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": 256 * 32 * 32, "max_pixels": max_pixels},
        enable_prefix_caching=True,
        seed=SEED,
    )
    out = {}

    def frame(name):
        if name == "holdout":
            ho = pd.read_csv(f"{run_dir}/holdout.csv")
            df = tr[tr.trace_id.isin(ho.trace_id)].reset_index(drop=True)
            split = "train"
        elif name == "train":
            df = tr.copy()
            split = "train"
        elif name == "unlabelled":
            df = pd.read_csv(f"{data}/unlabelled.csv")
            split = "unlabelled"
            if ids_csv:
                df = df[df.trace_id.isin(set(pd.read_csv(ids_csv).trace_id))].reset_index(
                    drop=True
                )  # only the rows we need
        else:
            df = te.copy()
            split = "test"
        have = df.image_file.map(lambda f: img_path(data, split, f) is not None)
        if have.mean() < 1:
            print(f"WARNING {name}: only {have.sum()}/{len(df)} images present", flush=True)
            df = df[have].reset_index(drop=True)
        return df, split

    def prompts(df, split, sys_t, user_fn):
        ps = []
        for _, r in df.iterrows():
            im = load_image(img_path(data, split, r.image_file), max_pixels)
            txt = processor.apply_chat_template(
                build_messages(sys_t, user_fn(r), im), tokenize=False, add_generation_prompt=True
            )
            ps.append({"prompt": txt, "multi_modal_data": {"image": im}})
        return ps

    for name in [s for s in splits if s]:  # '' used to fall through to the test set
        df, split = frame(name)
        if limit:
            df = df.head(limit)
        t0 = time.time()
        ps = prompts(df, split, DIAG_SYS, lambda r: diag_user_text(r, tax_block))
        greedy = llm.generate(ps, SamplingParams(temperature=0, max_tokens=8))
        beam_out = None
        if beams:
            try:
                from vllm.sampling_params import BeamSearchParams

                beam_out = llm.beam_search(ps, BeamSearchParams(beam_width=beams, max_tokens=8, length_penalty=0.0))
                print("beam search ok", flush=True)
            except Exception as e:
                print("vLLM beam search failed, using sampling:", repr(e)[:300], flush=True)
                beam_out = None
        samp = (
            llm.generate(ps, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=8, n=n_samples, seed=SEED))
            if (n_samples and beam_out is None)
            else None
        )
        rows = []
        for i, (_, r) in enumerate(df.iterrows()):
            cand = {}
            g = parse_code(greedy[i].outputs[0].text, valid)
            if beam_out is not None:
                for seq in beam_out[i].sequences:
                    c = parse_code(seq.text, valid)
                    if c:
                        cand[c] = cand.get(c, 0) + float(np.exp(seq.cum_logprob))
                if g and g not in cand:
                    cand[g] = max(cand.values(), default=1.0)
            else:
                if g:
                    cand[g] = cand.get(g, 0) + n_samples  # greedy counts as the strongest vote
                if samp:
                    for o in samp[i].outputs:
                        c = parse_code(o.text, valid)
                        if c:
                            cand[c] = cand.get(c, 0) + 1
            tot = sum(cand.values()) or 1.0
            cand = {k: v / tot for k, v in sorted(cand.items(), key=lambda kv: -kv[1])}
            rows.append(
                {
                    "trace_id": r.trace_id,
                    "pred": " ".join(list(cand)[:3]),
                    "probs": json.dumps(cand),
                    "raw": greedy[i].outputs[0].text[:40],
                }
            )
        res = pd.DataFrame(rows)
        res.to_csv(f"{run_dir}/vllm_diag_{name}.csv", index=False)
        print(
            f"{name}: {len(df)} rows in {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / max(1, len(df)):.2f}s/img)",
            flush=True,
        )
        if name == "holdout":
            m = res.merge(tr[["trace_id", "root_code"]], on="trace_id")
            s = 0.0
            for p, t in zip(m.pred.fillna("").str.split(), m.root_code):
                s += (1 / (p.index(t) + 1)) if t in p else 0
            out["holdout_map3"] = s / len(m)
            out["holdout_top1"] = float((m.pred.fillna("").str.split().str[0] == m.root_code).mean())
            print(f"HOLDOUT MAP@3 = {s / len(m):.4f} top1 {out['holdout_top1']:.3f} on {len(m)} rows", flush=True)
        out[f"n_{name}"] = len(df)

    for name in transcribe_splits:
        df, split = frame(name)
        if transcribe_limit:
            df = df.head(transcribe_limit)
        t0 = time.time()
        ps = prompts(df, split, TRANS_SYS, trans_user_text)
        gen = llm.generate(ps, SamplingParams(temperature=0, max_tokens=384, repetition_penalty=1.05))
        res = pd.DataFrame({"trace_id": df.trace_id, "ocr": [g.outputs[0].text for g in gen]})
        res.to_csv(
            f"{run_dir}/vllm_ocr_{name}{'_' + os.path.basename(ids_csv).split('.')[0] if ids_csv and name == 'unlabelled' else ''}.csv",
            index=False,
        )
        print(f"transcribe {name}: {len(df)} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if name in ("holdout", "train"):
            m = res.merge(tr[["trace_id", "transcription_steps"]], on="trace_id")
            sims = [difflib.SequenceMatcher(None, a, str(b)).ratio() for a, b in zip(m.ocr, m.transcription_steps)]
            out[f"ocr_sim_{name}"] = float(np.mean(sims))
            print(f"TRANSCRIPTION similarity {name}: {np.mean(sims):.3f}", flush=True)
    json.dump(out, open(f"{run_dir}/vllm_summary.json", "w"), indent=1)
    return out
