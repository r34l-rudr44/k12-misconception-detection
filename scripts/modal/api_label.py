"""Label photos of student work with a Claude model.

The competition host allowed external APIs for pseudo-labelling the unlabelled set (not at inference time), so this
is only ever run on train rows (to measure the labeller against gold labels) and on unlabelled rows. Never on test rows.

Tasks:
    diagnose   cached system prompt (instructions + full taxonomy) + per-row photo, question, answer key and the codes
               that other students' work on the same question received in train (excluding the rows being labelled).
               The reply ends with "ANSWER: CODE1, CODE2, CODE3". With transcribe=True the model first transcribes the
               working line by line.
    screen     read the student's final answer and compare it with the key ("FINAL: ... | MATCH: YES or NO"): a cheap
               filter for rows that probably contain a misconception.

Modes: sync (one request per row), submit / collect (Message Batches API, half price).
Spend is metered from `usage` into a ledger file and checked against a hard cap.
"""

import base64
import io
import json
import os
import re
import time

import pandas as pd
from PIL import Image, ImageOps

PRICES = {  # $ per 1M tokens: input, output (cache write 1.25x input, cache read 0.1x input); batch = 50%
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
LEDGER = "/models/api_ledger.json"

SYS_HEAD = """You are an expert mathematics teacher diagnosing Indonesian K-12 students' handwritten work.
You see a photo of one student's working, the question and the answer key. Identify the ROOT misconception: the
earliest conceptual error that drives the mistakes. Use exactly one code from the taxonomy below, or NONE when the work
is correct or has only minor slips (copying slips, small arithmetic slips that are not systematic). A student whose final
answer matches the key is almost always NONE.

Think briefly, then end your reply with one line in exactly this form, ranking your three most likely codes:
ANSWER: CODE1, CODE2, CODE3

Taxonomy (code | name | description):
"""


def taxonomy_text(tax):
    return (
        "\n".join(f"{r.id} | {r.name} | {r.description}" for r in tax.itertuples())
        + "\nNONE | No major misconception | Correct working, or only minor slips."
    )


def image_block(path, max_side=1568):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode(),
        },
    }


TRANSCRIBE = (
    "First transcribe the student's handwritten working exactly as written (keep their mistakes, do not fix anything), "
    "one written line per row in the form 'L1 | ...', 'L2 | ...', using plain-text math (fractions as a/b). "
    "Put it between a line 'TRANSCRIPTION:' and a line 'END TRANSCRIPTION'. Then diagnose the work."
)


def user_text(r, prior, transcribe=False):
    p = ", ".join(f"{c} ({n})" for c, n in prior) if prior else "none recorded"
    return (
        f"Grade {r.grade}. Topic: {r.topic_name}\nQuestion: {r.question_text}\nAnswer key: {r.correct_answer}\n"
        f"Codes other students received on this same question (count): {p}\n"
        + (TRANSCRIBE if transcribe else "Diagnose the student's work in the photo.")
    )


def transcript(text):
    m = re.search(r"TRANSCRIPTION:\s*\n(.*?)\n\s*END TRANSCRIPTION", text, re.S)
    return m.group(1).strip() if m else ""


def parse(text, valid):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if not m:
        return []
    out = []
    for tok in re.split(r"[,\s]+", m[-1].strip()):
        t = tok.strip(" .;*`").upper()
        for v in valid:
            if v.upper() == t and v not in out:
                out.append(v)
    return out[:3]


def cost(model, u, batch):
    pin, pout = PRICES[model]
    c = (
        u.get("input_tokens", 0) * pin
        + (u.get("cache_creation_input_tokens") or 0) * pin * 1.25
        + (u.get("cache_read_input_tokens") or 0) * pin * 0.1
        + u.get("output_tokens", 0) * pout
    ) / 1e6
    return c * (0.5 if batch else 1.0)


def ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"spent": 0.0, "runs": []}


SCREEN_SYS = (
    "You check Indonesian K-12 students' handwritten math work. From the photo, read the student's FINAL answer "
    "(the last result they wrote for the question) and compare it with the answer key. Treat equivalent forms as equal "
    "(3/4 = 0.75 = 0,75; 1.500 = 1500; units optional). Reply with exactly one line:\nFINAL: <student's final answer> | MATCH: YES or NO"
)


def screen_parse(text):
    m = re.findall(r"MATCH:\s*(YES|NO)", text.upper())
    return m[-1] if m else ""


def main(
    data,
    out_dir,
    rows_csv,
    model,
    mode="sync",
    n=20,
    cap_total=10.0,
    effort="medium",
    max_tokens=6000,
    batch_id="",
    est_per_row=0.0,
    exclude_ids_for_prior=(),
    transcribe=False,
    task="diagnose",
    chunk=750,
    max_side=1568,
):
    import anthropic

    client = anthropic.Anthropic()
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(f"{data}/train.csv")
    tax = pd.read_csv(f"{data}/taxonomy_v1.csv")
    valid = list(tax.id) + ["NONE"]
    system = [{"type": "text", "text": SYS_HEAD + taxonomy_text(tax), "cache_control": {"type": "ephemeral"}}]
    if task == "screen":
        system = SCREEN_SYS
    rows = pd.read_csv(rows_csv)
    if n:
        rows = rows.head(n)
    split = "train" if rows.trace_id.isin(tr.trace_id).all() else "unlabelled"
    excl = set(rows.trace_id) | set(exclude_ids_for_prior)
    ref = tr[~tr.trace_id.isin(excl)]
    prior = {q: list(g.root_code.value_counts().head(8).items()) for q, g in ref.groupby("question_id")}
    L = ledger()

    def params(r):
        return dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": effort},
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_block(f"{data}/images/{split}/{r.image_file}", max_side),
                        {
                            "type": "text",
                            "text": (
                                f"Question: {r.question_text}\nAnswer key: {r.correct_answer}"
                                if task == "screen"
                                else user_text(r, prior.get(r.question_id), transcribe)
                            ),
                        },
                    ],
                }
            ],
        )

    def record(rid, msg, batch):
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage.to_dict() if hasattr(msg.usage, "to_dict") else dict(msg.usage)
        return {
            "trace_id": rid,
            "codes": " ".join(parse(text, valid)),
            "stop": msg.stop_reason,
            "cost": cost(model, u, batch),
            "in": u.get("input_tokens", 0),
            "cache_r": u.get("cache_read_input_tokens") or 0,
            "out": u.get("output_tokens", 0),
            "tail": text[-300:],
            "transcript": transcript(text),
            "match": screen_parse(text) if task == "screen" else "",
        }

    tag = (
        f"{model}_{os.path.basename(rows_csv).split('.')[0]}"
        + ("_tr" if transcribe else "")
        + ("_screen" if task == "screen" else "")
    )
    if mode == "sync":
        recs = []
        for r in rows.itertuples():
            if L["spent"] >= cap_total:
                print("CAP reached, stopping", flush=True)
                break
            try:
                msg = client.messages.create(**params(r))
            except anthropic.APIStatusError as e:
                print("API error", r.trace_id, e.status_code, str(e)[:200], flush=True)
                continue
            rec = record(r.trace_id, msg, False)
            recs.append(rec)
            L["spent"] += rec["cost"]
        res = pd.DataFrame(recs)
    elif mode == "submit":
        est = est_per_row * len(rows)
        if not est_per_row or L["spent"] + est > cap_total:
            return {"refused": True, "spent": L["spent"], "estimate": est, "cap": cap_total}
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        ids = []
        for i in range(0, len(rows), chunk):  # Batches API request cap is 256 MB, so split large jobs
            part = rows.iloc[i : i + chunk]
            reqs = [
                Request(custom_id=r.trace_id, params=MessageCreateParamsNonStreaming(**params(r)))
                for r in part.itertuples()
            ]
            b = client.messages.batches.create(requests=reqs)
            ids.append(b.id)
            pe = est_per_row * len(part)
            L["runs"].append({"batch": b.id, "model": model, "rows": len(part), "estimate": pe, "time": time.time()})
            L["reserved"] = L.get("reserved", 0) + pe
            json.dump(L, open(LEDGER, "w"), indent=1)
            print("submitted", b.id, len(part), flush=True)
        return {"batch_id": ",".join(ids), "rows": len(rows), "estimate": est, "spent": L["spent"]}
    else:  # collect
        recs = []
        bids = batch_id.split(",")
        for bid in bids:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                return {
                    "status": b.processing_status,
                    "batch": bid,
                    "counts": b.request_counts.to_dict()
                    if hasattr(b.request_counts, "to_dict")
                    else str(b.request_counts),
                }
        for bid in bids:
            part = []
            for x in client.messages.batches.results(bid):
                if x.result.type == "succeeded":
                    part.append(record(x.custom_id, x.result.message, True))
                else:
                    part.append({"trace_id": x.custom_id, "codes": "", "stop": x.result.type, "cost": 0.0})
            pc = float(sum(r["cost"] for r in part))
            L["spent"] += pc
            recs += part
            for run in L["runs"]:
                if run.get("batch") == bid and "actual" not in run:
                    L["reserved"] = L.get("reserved", 0) - run["estimate"]
                    run["actual"] = pc
        res = pd.DataFrame(recs)
        tag = f"{tag}_batch"
    json.dump(L, open(LEDGER, "w"), indent=1)
    res.to_csv(f"{out_dir}/api_{tag}.csv", index=False)
    summ = {
        "rows": len(res),
        "cost": round(float(res.cost.sum()), 4),
        "per_row": round(float(res.cost.mean()), 5) if len(res) else 0,
        "spent_total": round(L["spent"], 4),
        "parsed": float((res.codes.fillna("") != "").mean()) if len(res) else 0,
        "mean_out_tokens": float(res["out"].mean()) if "out" in res and len(res) else 0,
    }
    if transcribe and "transcript" in res:
        summ["transcribed"] = float((res.transcript.fillna("") != "").mean())
    if split == "train" and len(res):
        y = res.merge(tr[["trace_id", "root_code"]], on="trace_id")
        s = 0.0
        for c, t in zip(y.codes.fillna("").str.split(), y.root_code):
            s += 1 / (c.index(t) + 1) if t in c else 0
        summ["map3"] = round(s / len(y), 4)
        summ["top1"] = round(float((y.codes.fillna("").str.split().str[0] == y.root_code).mean()), 4)
    print(summ, flush=True)
    return summ
