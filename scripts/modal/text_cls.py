"""Text-only sequence classifier: Qwen3 + LoRA + linear head over (question, answer key, transcribed student work).

One call trains one fold (fold = -1 trains on all rows) and writes validation and test probabilities over the 76
classes as CSV files for blending. Training rows show the reader's OCR, the public OCR or the human transcription at
random, so the model learns from the same kind of noisy text it gets at test time. Training checkpoints every 100
optimizer steps and resumes, because a preempted cloud GPU job is restarted from the beginning.
"""

import json
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch


def build_text(r, work):
    return (
        f"Grade {r.grade}. Topic: {r.topic_name}\nQuestion: {r.question_text}\nAnswer key: {r.correct_answer}\n"
        f"Student work:\n{work}"
    )


def _s(x):
    return x if isinstance(x, str) else ""


_LINE = re.compile(r"^\s*L\d+\s*\|")
_META = re.compile(r"\[student answered|<\s*/?\s*mis", re.I)


def clean_gold(x):
    """Gold transcriptions sometimes carry annotator notes ("[student answered the question correctly]", <misconseption>
    tags) that give the label away and never occur in test OCR. Keep only real 'L<n> | ...' lines without such notes."""
    keep = [l for l in _s(x).splitlines() if _LINE.match(l) and not _META.search(l)]
    return "\n".join(keep)


def map3(P, y):
    top = np.argsort(-P, 1)[:, :3]
    return float(np.mean((top == y[:, None]) @ np.array([1, 1 / 2, 1 / 3])))


def load_tok(base):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"  # the student work and its final answer are at the END of the text: never cut them
    return tok


@torch.no_grad()
def predict_probs(model, tok, texts, n_classes, max_len, bs=16):
    """Class probabilities for a list of texts. Longest first, so an out-of-memory error shows up on the first batch;
    on OOM the batch size is halved and the batch retried."""
    model.eval()
    P = np.zeros((len(texts), n_classes), dtype=np.float32)
    order = np.argsort([-len(t) for t in texts])
    s = 0
    while s < len(texts):
        idx = order[s : s + bs]
        try:
            enc = tok(
                [texts[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt"
            ).to(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
            s += bs
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"OOM at inference -> batch {bs}", flush=True)
    return P


def sanity(P, classes, name):
    """Cheap guard against a silently broken model (e.g. head not restored -> near-uniform probabilities)."""
    top = P.argmax(1)
    none = float((top == classes.index("NONE")).mean())
    conf = float(P.max(1).mean())
    print(
        f"SANITY {name}: mean max-prob {conf:.3f} | NONE top-1 share {none:.3f} | distinct top-1 codes {len(set(top))}",
        flush=True,
    )
    assert len(P) < 500 or (conf > 0.25 and 0.1 < none < 0.7 and len(set(top)) >= 15), (
        f"{name}: predictions look broken"
    )
    return {"conf": round(conf, 3), "none_share": round(none, 3)}


def main(
    base,
    data_dir,
    out,
    fold=0,
    epochs=2.0,
    lr=2e-4,
    lora_r=16,
    batch=4,
    accum=2,
    max_len=640,
    seed=42,
    p_gold=0.25,
    p_pub=0.25,
    limit=0,
    tag="",
    extra="",
    skip_test=False,
    infer_len=1024,
    on_save=None,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    tag = tag or f"fold{fold}"
    on_save = on_save or (
        lambda: None
    )  # called after every file write (Modal: volume commit), so a later crash loses nothing
    tr = pd.read_csv(f"{data_dir}/text_train.csv")
    te = pd.read_csv(f"{data_dir}/text_test.csv")
    classes = json.load(open(f"{data_dir}/classes.json"))
    ci = {c: i for i, c in enumerate(classes)}
    tr["gold"] = tr.gold.map(clean_gold)
    trn = tr[tr.fold != fold].reset_index(drop=True)
    val = tr[tr.fold == fold].reset_index(drop=True)
    if extra:  # pseudo-labelled rows (Sonnet label + transcript), training only
        ex = pd.read_csv(f"{data_dir}/{extra}.csv")
        ex = ex[ex.root_code.isin(ci)]
        trn = pd.concat([trn, ex], ignore_index=True)
        print(f"extra {extra}: +{len(ex)} rows", flush=True)
    if limit:
        trn, val, te = trn.head(limit), val.head(min(limit, len(val))), te.head(limit)
    tok = load_tok(base)
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(classes), dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False  # no KV cache: classification only
    torch.nn.init.zeros_(model.score.weight)  # start at uniform probs (random head gave loss ~10)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="SEQ_CLS",
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    print(f"loaded in {time.time() - t0:.0f}s; train {len(trn)} val {len(val)} test {len(te)}", flush=True)

    def pick(r):
        u = random.random()
        if u < p_gold and _s(r.gold):
            return r.gold
        if u < p_gold + p_pub and _s(r.ocr_pub):
            return r.ocr_pub
        return _s(r.ocr_v1) or _s(r.ocr_pub)

    total = int(len(trn) * epochs)
    order = []
    while len(order) < total:
        order += list(np.random.permutation(len(trn)))
    micro = [order[i : i + batch] for i in range(0, total, batch)]
    n_steps = math.ceil(len(micro) / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * n_steps)), n_steps)
    model.train()
    t0 = time.time()
    losses = []
    step = 0
    # Modal can preempt a GPU container at any time and restarts the call from scratch: checkpoint and resume.
    ck = f"{out}/{tag}_ckpt.pt"
    k0 = 0
    sig = [base, fold, epochs, lr, lora_r, batch, accum, max_len, seed, p_gold, p_pub, extra, len(trn), len(micro)]
    if os.path.exists(ck):
        try:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            named = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert st["sig"] == sig, "checkpoint belongs to a different run"
            assert set(named) == set(st["params"]), "parameter names differ"
            opt.load_state_dict(st["opt"])
            sch.load_state_dict(st["sch"])
            for n, p in named.items():
                p.data.copy_(st["params"][n].to(p.device, p.dtype))
            k0, step, losses = st["k"], st["step"], st["losses"]
            print(f"RESUMED from checkpoint at step {step}/{n_steps}", flush=True)
            del st
        except Exception as e:
            print("checkpoint not usable, starting fresh:", repr(e)[:300], flush=True)
    step0 = step
    for k, idx in enumerate(micro):
        if k < k0:
            continue
        rows = [trn.iloc[i] for i in idx]
        enc = tok(
            [build_text(r, pick(r)) for r in rows],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(0)
        labels = torch.tensor([ci[r.root_code] for r in rows], device=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        (loss / accum).backward()
        losses.append(loss.item())
        if (k + 1) % accum == 0 or k == len(micro) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == n_steps:
                el = time.time() - t0
                sps = el / max(1, step - step0)
                print(
                    f"step {step}/{n_steps} loss {np.mean(losses[-25 * accum :]):.3f} {sps:.2f}s/step eta {(n_steps - step) * sps / 60:.1f} min",
                    flush=True,
                )
            if step % 100 == 0 or step == n_steps:  # the last one lets a restart during inference skip the training
                try:
                    torch.save(
                        {
                            "k": k + 1,
                            "step": step,
                            "losses": losses,
                            "sig": sig,
                            "opt": opt.state_dict(),
                            "sch": sch.state_dict(),
                            "params": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                        },
                        ck + ".tmp",
                    )
                    os.replace(ck + ".tmp", ck)
                    on_save()
                except Exception as e:
                    print("checkpoint save failed:", repr(e)[:300], flush=True)
    train_min = (time.time() - t0) / 60
    model.save_pretrained(f"{out}/{tag}_adapter")
    on_save()  # before inference, so an inference failure keeps the training
    del opt, sch
    [setattr(p_, "grad", None) for p_ in params]
    torch.cuda.empty_cache()

    def predict(df, col):
        return predict_probs(
            model, tok, [build_text(r, _s(getattr(r, col))) for r in df.itertuples()], len(classes), infer_len
        )

    def save(df, P, name):
        d = pd.DataFrame(P.round(5), columns=classes)
        d.insert(0, "trace_id", df.trace_id.values)
        d.to_csv(f"{out}/{tag}_{name}.csv", index=False)
        on_save()

    res = {"fold": fold, "train_min": round(train_min, 1), "final_loss": float(np.mean(losses[-50:]))}
    t1 = time.time()
    if len(val):
        y = np.array([ci[c] for c in val.root_code])
        Pv = {c: predict(val, c) for c in ["ocr_v1", "ocr_pub"]}
        Pv["avg"] = (Pv["ocr_v1"] + Pv["ocr_pub"]) / 2
        for c, P in Pv.items():
            res[f"val_map3_{c}"] = round(map3(P, y), 4)
            save(val, P, f"val_{c}")
        res["val_top1_avg"] = round(float((Pv["avg"].argmax(1) == y).mean()), 4)
        print("VAL", {k: v for k, v in res.items() if k.startswith("val")}, flush=True)
    if not skip_test:
        for c in ["ocr_v1", "ocr_pub"]:
            P = predict(te, c)
            res[f"sanity_{c}"] = sanity(P, classes, f"test {c}")
            save(te, P, f"test_{c}")
    res["infer_min"] = round((time.time() - t1) / 60, 1)
    json.dump(res, open(f"{out}/{tag}_result.json", "w"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)  # finished: nothing left to resume
    on_save()
    print(res, flush=True)
    return res


def predict_only(base, data_dir, out, adapter, tag, max_len=1024, bs=16, check_fold=-1, on_save=None):
    """Load a saved LoRA adapter (+ score head) and write test probabilities for both OCR sources.
    check_fold >= 0: first score that fold's validation rows (must reproduce the MAP@3 from training time for a fold
    adapter). check_fold = -1: score 200 train rows instead (in-sample for a full-data adapter, so it must be high).
    Either way a broken load is caught in the first minute, before the test set is predicted."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    assert os.path.isdir(adapter), f"adapter not found: {adapter}"  # fail before the 16 GB base model is loaded
    on_save = on_save or (lambda: None)
    te = pd.read_csv(f"{data_dir}/text_test.csv")
    tr = pd.read_csv(f"{data_dir}/text_train.csv")
    classes = json.load(open(f"{data_dir}/classes.json"))
    ci = {c: i for i, c in enumerate(classes)}
    tok = load_tok(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(classes), dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False  # no KV cache: classification only
    model = PeftModel.from_pretrained(model, adapter).eval()
    t0 = time.time()
    head = [p for n, p in model.named_parameters() if "score" in n and "modules_to_save" in n]
    assert head and float(head[0].float().abs().sum()) > 0, "classification head was not restored from the adapter"
    texts = lambda df, col: [build_text(r, _s(getattr(r, col))) for r in df.itertuples()]
    chk = tr[tr.fold == check_fold] if check_fold >= 0 else tr.sample(200, random_state=0)
    Pc = sum(predict_probs(model, tok, texts(chk, c), len(classes), max_len, bs) for c in ["ocr_v1", "ocr_pub"]) / 2
    res = {
        "tag": tag,
        "rows": len(te),
        "check_rows": len(chk),
        "check_map3": round(map3(Pc, np.array([ci[c] for c in chk.root_code])), 4),
    }
    print("CHECK", res, flush=True)
    assert res["check_map3"] > 0.5, "adapter scores too low on the check rows: wrong or broken adapter"
    for col in ["ocr_v1", "ocr_pub"]:
        P = predict_probs(model, tok, texts(te, col), len(classes), max_len, bs)
        res[f"sanity_{col}"] = sanity(P, classes, f"test {col}")
        d = pd.DataFrame(P.round(5), columns=classes)
        d.insert(0, "trace_id", te.trace_id.values)
        d.to_csv(f"{out}/{tag}_test_{col}.csv", index=False)
        on_save()
    res["infer_min"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(f"{out}/{tag}_result.json", "w"), indent=1)
    on_save()
    print(res, flush=True)
    return res
