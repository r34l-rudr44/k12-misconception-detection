"""LoRA fine-tune of Qwen3-VL on two tasks from the same photos:

    diagnose     photo + question + answer key -> "ROOT: <code>" and a one-line explanation
    transcribe   photo -> the student's working, line by line ("L1 | ...")

The transcription task is what the rest of the pipeline uses (the model becomes the "reader"). A stratified 20% of
train is held out and written to <out>/holdout.csv. Runs inside the Modal GPU image (see modal_app.py::train_vlm).
"""

import json
import math
import os
import random
import time

import numpy as np
import torch

from common import (
    DIAG_SYS,
    SEED,
    TRANS_SYS,
    build_messages,
    diag_target,
    diag_user_text,
    holdout_split,
    img_path,
    load_image,
    load_tables,
    taxonomy_block,
    trans_user_text,
)


def main(
    base,
    data,
    out,
    epochs=2.0,
    limit=0,
    max_pixels=768 * 768,
    lr=1e-4,
    holdout_frac=0.2,
    batch=2,
    accum=4,
    lora_r=32,
    tax_in_prompt=True,
    tasks=("diagnose", "transcribe"),
    none_keep=1.0,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(out, exist_ok=True)
    tr, te, tax = load_tables(data)
    tax_block = taxonomy_block(tax) if tax_in_prompt else ""
    trn, ho = holdout_split(tr, holdout_frac)
    have = trn.image_file.map(lambda f: img_path(data, "train", f) is not None)
    if have.mean() < 1:
        print(f"WARNING: only {have.sum()}/{len(trn)} train images present; training on those", flush=True)
        trn = trn[have].reset_index(drop=True)
    if limit:
        trn = trn.head(limit)
    ho[["trace_id", "root_code"]].to_csv(f"{out}/holdout.csv", index=False)
    print(f"train rows {len(trn)} holdout {len(ho)} tasks {tasks}", flush=True)

    processor = AutoProcessor.from_pretrained(base)
    ip = processor.image_processor
    ip.size = {"shortest_edge": 256 * 32 * 32, "longest_edge": max_pixels}
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    # ---- samples: (sys, user_text, image_path, answer)
    samples = []
    rng = random.Random(SEED)
    for _, r in trn.iterrows():
        p = img_path(data, "train", r.image_file)
        if "diagnose" in tasks and (r.root_code != "NONE" or rng.random() < none_keep):
            samples.append((DIAG_SYS, diag_user_text(r, tax_block), p, diag_target(r)))
        if "transcribe" in tasks and isinstance(r.transcription_steps, str):
            samples.append((TRANS_SYS, trans_user_text(r), p, r.transcription_steps[:1500]))
    print("samples", len(samples), flush=True)

    def encode(batch_samples):
        texts, imgs, plens = [], [], []
        for sys_t, user_t, p, ans in batch_samples:
            im = load_image(p, max_pixels)
            full = processor.apply_chat_template(
                build_messages(sys_t, user_t, im, ans), tokenize=False, add_generation_prompt=False
            )
            prompt = processor.apply_chat_template(
                build_messages(sys_t, user_t, im), tokenize=False, add_generation_prompt=True
            )
            texts.append(full)
            imgs.append(im)
            plens.append(prompt)
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        penc = processor(text=plens, images=imgs, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i in range(len(batch_samples)):
            pl = int(penc["attention_mask"][i].sum())
            labels[i, :pl] = -100
        enc["labels"] = labels
        return enc

    steps_per_epoch = math.ceil(len(samples) / batch)
    total_steps = int(steps_per_epoch * epochs)
    opt_steps = max(1, total_steps // accum)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.999)
    )
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * opt_steps), opt_steps)
    model.train()
    t0 = time.time()
    step = 0
    losses = []
    log = open(f"{out}/train_log.txt", "a")
    while step < total_steps:
        random.shuffle(samples)
        for b in range(0, len(samples), batch):
            if step >= total_steps:
                break
            enc = encode(samples[b : b + batch])
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss / accum
            loss.backward()
            losses.append(loss.item() * accum)
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 20 == 0:
                el = time.time() - t0
                msg = f"step {step}/{total_steps} loss {np.mean(losses[-20:]):.4f} {el / step:.2f}s/step eta {(total_steps - step) * el / step / 60:.1f}min mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
    model.save_pretrained(f"{out}/adapter")
    processor.save_pretrained(f"{out}/adapter")
    meta = {
        "base": base,
        "epochs": epochs,
        "samples": len(samples),
        "steps": total_steps,
        "max_pixels": max_pixels,
        "lr": lr,
        "lora_r": lora_r,
        "tax_in_prompt": tax_in_prompt,
        "tasks": list(tasks),
        "final_loss": float(np.mean(losses[-50:])),
        "minutes": (time.time() - t0) / 60,
    }
    json.dump(meta, open(f"{out}/train_meta.json", "w"), indent=1)
    print(meta, flush=True)
    return meta
