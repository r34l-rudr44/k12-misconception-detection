"""CPU-only pipeline: priors and a TF-IDF logistic regression on OCR text.

Components (each gives an (n, K) probability matrix over the 75 codes + NONE):
    q       hierarchical prior: question -> topic -> global, with smoothing
    topic   the topic-level prior on its own
    lr      TF-IDF (word and character n-grams) logistic regression on [topic || question || key || OCR text],
            plus answer-match features and the grade
    qa      (question, student's final answer) prior, smoothed towards the question prior

The blend weights and a NONE gate driven by the answer match are tuned on 5-fold out-of-fold MAP@3.

Usage:
    python scripts/cpu_baseline.py [ocr dir] [ocr tag]
    python scripts/cpu_baseline.py                    # public OCR: data/external/baseline_ocr, tag qwen3.5-9b
    python scripts/cpu_baseline.py work/v1ocr v1      # the fine-tuned reader's OCR

The OCR directory holds ocr_train_<tag>.csv and ocr_test_<tag>.csv (columns trace_id, ocr).

Pseudo-labelled rows (training side only, never evaluated on) are added with environment variables:
    K12_PSEUDO       name of a set in scripts/modal/textdata/ (see prep_pseudo.py)
    K12_PSEUDO_W     weight of a pseudo row in the priors (default 0.5)
    K12_PSEUDO_COL   text column used for the pseudo rows (default ocr_v1)

Writes work/cpu_{oof,test}_<tag>.npy, work/cpu_meta_<tag>.json and work/submission_cpu_<tag>.csv.
"""

import ast
import json
import operator
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "work")
os.makedirs(OUT, exist_ok=True)
OCR_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "external", "baseline_ocr")
OCR_TAG = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5-9b"
SEED, N_FOLDS = 42, 5

train = pd.read_csv(f"{D}/train.csv")
test = pd.read_csv(f"{D}/test.csv")
tax = pd.read_csv(f"{D}/taxonomy_v1.csv")
ocr_tr = pd.read_csv(f"{OCR_DIR}/ocr_train_{OCR_TAG}.csv")
ocr_te = pd.read_csv(f"{OCR_DIR}/ocr_test_{OCR_TAG}.csv")
train = train.merge(ocr_tr, on="trace_id", how="left")
test = test.merge(ocr_te, on="trace_id", how="left")
train["ocr"] = train["ocr"].fillna("")
test["ocr"] = test["ocr"].fillna("")

CLASSES = sorted(set(train.root_code) | set(tax.id) | {"NONE"})
C2I = {c: i for i, c in enumerate(CLASSES)}
K = len(CLASSES)
y = train.root_code.map(C2I).values
yt = train.root_code.values
train["_w"] = 1.0

# Optional pseudo-labelled rows from the unlabelled set (training side only, never evaluated on)
PSEUDO = os.environ.get("K12_PSEUDO", "")
PW = float(os.environ.get("K12_PSEUDO_W", 0.5))
ps = None
if PSEUDO:
    ps = pd.read_csv(f"{ROOT}/scripts/modal/textdata/{PSEUDO}.csv")
    ps = ps[ps.root_code.isin(C2I)].reset_index(drop=True)
    ps["question_id"] = ps.trace_id.map(pd.read_csv(f"{D}/unlabelled.csv").set_index("trace_id").question_id)
    ps["ocr"] = ps[os.environ.get("K12_PSEUDO_COL", "ocr_v1")].fillna("").astype(str)
    ps["_w"] = PW
    yps = ps.root_code.map(C2I).values
    print(f"pseudo rows: {len(ps)} from {PSEUDO} (prior weight {PW})")


def with_ps(df):
    return pd.concat([df, ps], ignore_index=True) if ps is not None else df


def map3(y_true, top3):
    s = 0.0
    for t, preds in zip(y_true, top3):
        seen = []
        for k, p in enumerate(preds[:3]):
            if p in seen:
                continue
            seen.append(p)
            if p == t:
                s += 1.0 / (k + 1)
                break
    return s / len(y_true)


def top3(P):
    return [[CLASSES[j] for j in row] for row in np.argsort(-P, axis=1)[:, :3]]


# ---------------- priors ----------------
def fit_priors(df, a, b):
    df = with_ps(df)
    cnt = lambda d: np.bincount(d.root_code.map(C2I), weights=d._w, minlength=K)
    g = cnt(df) + 0.5
    g /= g.sum()
    topic, ques = {}, {}
    for t, d in df.groupby("topic_name"):
        n = cnt(d)
        topic[t] = (n + b * g) / (n.sum() + b)
    for q, d in df.groupby("question_id"):
        n = cnt(d)
        pt = topic.get(d.topic_name.iloc[0], g)
        ques[q] = (n + a * pt) / (n.sum() + a)
    return g, topic, ques


def predict_priors(df, pr):
    g, topic, ques = pr
    PQ = np.zeros((len(df), K))
    PT = np.zeros((len(df), K))
    for i, (q, t) in enumerate(zip(df.question_id, df.topic_name)):
        PT[i] = topic.get(t, g)
        PQ[i] = ques.get(q, PT[i])
    return PQ, PT


skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
# stratify on root_code but singletons break StratifiedKFold -> group rare codes
strat = train.root_code.where(train.root_code.map(train.root_code.value_counts()) >= N_FOLDS, "RARE")
FOLDS = list(skf.split(train, strat))

best = None
for a in [0.5, 1, 2, 4, 8]:
    for b in [1, 3, 10]:
        oof = np.zeros((len(train), K))
        for tr, va in FOLDS:
            oof[va] = predict_priors(train.iloc[va], fit_priors(train.iloc[tr], a, b))[0]
        sc = map3(yt, top3(oof))
        if best is None or sc > best[0]:
            best = (sc, a, b)
print("Prior CV MAP@3 = %.4f (a=%s, b=%s)" % best)
A, B = best[1], best[2]
oof_q = np.zeros((len(train), K))
oof_t = np.zeros((len(train), K))
for tr, va in FOLDS:
    oof_q[va], oof_t[va] = predict_priors(train.iloc[va], fit_priors(train.iloc[tr], A, B))
PQ_te, PT_te = predict_priors(test, fit_priors(train, A, B))


# ---------------- text normalisation / answer match ----------------
def strip_lines(t):
    return "\n".join(re.sub(r"^\s*L\d+\s*\|\s*", "", l) for l in str(t).splitlines())


def norm_ans(s):
    s = str(s).lower()
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(left|right|text|mathrm|displaystyle|,|;|!)", "", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace(":", "/").replace("−", "-")
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("√", "sqrt")
    s = re.sub(r"[\s\$\{\}\\]", "", s)
    s = re.sub(r"^(jadi|jawab(an)?|maka|hasil(nya)?|=)+", "", s)
    s = re.sub(r"^[a-z]\=", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)
    s = s.replace(",", ".")
    s = re.sub(r"(cm|m|kg|g|liter|l|rp|buah|orang|hari|jam|menit|derajat|°|%)+$", "", s)
    return s.strip(".")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _arith(s):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(s)

    return ev(ast.parse(s, mode="eval").body)


def to_num(s):
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            return _arith(a) / _arith(b)
        return float(s)
    except Exception:
        return None


def answer_feats(text, key):
    lines = [l for l in strip_lines(text).splitlines() if l.strip()]
    if not lines:
        return [0, 0, 1, 0]
    k = norm_ans(key)
    kn = to_num(k)
    tail = [norm_ans(l) for l in lines[-3:]]
    match = int(bool(k) and any(k == t or t.endswith("=" + k) or t.endswith(k) for t in tail))
    anywhere = int(bool(k) and len(k) >= 2 and any(k in norm_ans(l) for l in lines))
    if not match and kn is not None:
        for t in tail[::-1]:
            nums = re.findall(r"-?\d+(?:\.\d+)?", t)
            if nums and abs(float(nums[-1]) - kn) < 1e-6:
                match = 1
                break
    return [match, min(len(lines), 30) / 30, 0, anywhere]


def student_answer(text):
    lines = [l for l in strip_lines(text).splitlines() if l.strip()]
    if not lines:
        return ""
    a = norm_ans(lines[-1]).split("=")[-1]
    v = to_num(a)
    return f"{v:.6g}" if v is not None else a[:30]


def make_text(df):
    return (
        df.topic_name.astype(str)
        + " || "
        + df.question_text.astype(str).str[:400]
        + " || KEY "
        + df.correct_answer.astype(str)
        + " || "
        + df.ocr.map(strip_lines)
    )


train["_txt"] = make_text(train).fillna("").astype(str)
test["_txt"] = make_text(test).fillna("").astype(str)
A_tr = np.array([answer_feats(t, k) for t, k in zip(train.ocr, train.correct_answer)], float)
A_te = np.array([answer_feats(t, k) for t, k in zip(test.ocr, test.correct_answer)], float)
m = A_tr[:, 0].astype(bool)
isnone = (train.root_code == "NONE").values
print(
    "answer-match on OCR: precision for NONE %.3f, recall %.3f, match rate train %.3f test %.3f"
    % (isnone[m].mean(), m[isnone].mean(), m.mean(), A_te[:, 0].mean())
)


# ---------------- TF-IDF + LR ----------------
def build_X(fit_txt, txts, A, grades):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True).fit(fit_txt)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True).fit(
        fit_txt
    )
    out = []
    for t, a, g in zip(txts, A, grades):
        G = np.eye(13)[np.clip(g.astype(int), 0, 12)]
        out.append(hstack([w.transform(t), c.transform(t), csr_matrix(a * 3.0), csr_matrix(G)]).tocsr())
    return out


def fit_predict_lr(Xtr, ytr, Xva, C=4.0):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, ytr)
    P = np.zeros((Xva.shape[0], K))
    P[:, clf.classes_] = clf.predict_proba(Xva)
    return P


fit_txt = pd.concat([train._txt, test._txt])

if ps is not None:
    ps["_txt"] = make_text(ps).fillna("").astype(str)
    A_ps = np.array([answer_feats(t, k) for t, k in zip(ps.ocr, ps.correct_answer)], float)
    X_tr, X_te, X_ps = build_X(
        fit_txt,
        [train._txt, test._txt, ps._txt],
        [A_tr, A_te, A_ps],
        [train.grade.values, test.grade.values, ps.grade.fillna(0).values],
    )
    plus = lambda X, yy: (vstack([X, X_ps]).tocsr(), np.concatenate([yy, yps]))
else:
    X_tr, X_te = build_X(fit_txt, [train._txt, test._txt], [A_tr, A_te], [train.grade.values, test.grade.values])
    plus = lambda X, yy: (X, yy)
oof_lr = np.zeros((len(train), K))
for tr, va in FOLDS:
    oof_lr[va] = fit_predict_lr(*plus(X_tr[tr], y[tr]), X_tr[va])
P_lr_te = fit_predict_lr(*plus(X_tr, y), X_te)
print(
    "CV MAP@3 LR text: %.4f | Q-prior: %.4f | topic: %.4f"
    % (map3(yt, top3(oof_lr)), map3(yt, top3(oof_q)), map3(yt, top3(oof_t)))
)

# ---------------- (question, answer) prior ----------------
ans_tr = np.array([student_answer(t) for t in train.ocr])
ans_te = np.array([student_answer(t) for t in test.ocr])
ans_ps = np.array([student_answer(t) for t in ps.ocr]) if ps is not None else None


def fit_qa(q_ids, ans, yy):
    cnt = {}
    for q, a, c in zip(q_ids, ans, yy):
        if a:
            cnt.setdefault((q, a), np.zeros(K))[c] += 1
    if ps is not None:
        for q, a, c in zip(ps.question_id.values, ans_ps, yps):
            if a:
                cnt.setdefault((q, a), np.zeros(K))[c] += PW
    return cnt


def predict_qa(q_ids, ans, cnt, P_q, lam):
    P = P_q.copy()
    for i, (q, a) in enumerate(zip(q_ids, ans)):
        n = cnt.get((q, a))
        if n is not None:
            P[i] = (n + lam * P_q[i]) / (n.sum() + lam)
    return P


best_qa = None
for lam in [0.5, 1, 2, 4, 8]:
    oof = np.zeros((len(train), K))
    for tr, va in FOLDS:
        oof[va] = predict_qa(
            train.question_id.values[va],
            ans_tr[va],
            fit_qa(train.question_id.values[tr], ans_tr[tr], y[tr]),
            oof_q[va],
            lam,
        )
    sc = map3(yt, top3(oof))
    if best_qa is None or sc > best_qa[0]:
        best_qa = (sc, lam, oof)
oof_qa = best_qa[2]
P_qa_te = predict_qa(test.question_id.values, ans_te, fit_qa(train.question_id.values, ans_tr, y), PQ_te, best_qa[1])
print("CV MAP@3 Q+answer prior: %.4f (lambda=%s)" % (best_qa[0], best_qa[1]))


# ---------------- NONE gate ----------------
def gate(P, A, w):
    P = P.copy()
    i = C2I["NONE"]
    P[:, i] = P[:, i] * (1 + w * A[:, 0])
    return P / P.sum(1, keepdims=True)


# ---------------- blend ----------------
def weight_grid(n, steps=10):
    if n == 1:
        yield (steps,)
        return
    for i in range(steps + 1):
        for rest in weight_grid(n - 1, steps - i):
            yield (i,) + rest


comps = {"lr": (oof_lr, P_lr_te), "q": (oof_q, PQ_te), "topic": (oof_t, PT_te), "qa": (oof_qa, P_qa_te)}
names = list(comps)
bestb = (-1, None, None)
for g in weight_grid(len(comps)):
    w = np.array(g) / 10
    P = sum(w[i] * comps[n][0] for i, n in enumerate(names))
    for gw in [0, 1, 2, 4, 8]:
        sc = map3(yt, top3(gate(P, A_tr, gw)))
        if sc > bestb[0]:
            bestb = (sc, w, gw)
sc, w, gw = bestb
print("Blend CV MAP@3 = %.4f weights %s gate %s" % (sc, dict(zip(names, w)), gw))
P_oof = gate(sum(w[i] * comps[n][0] for i, n in enumerate(names)), A_tr, gw)
P_te = gate(sum(w[i] * comps[n][1] for i, n in enumerate(names)), A_te, gw)

np.save(f"{OUT}/cpu_oof_{OCR_TAG}.npy", P_oof)
np.save(f"{OUT}/cpu_test_{OCR_TAG}.npy", P_te)
json.dump(
    {"classes": CLASSES, "weights": dict(zip(names, map(float, w))), "gate": gw, "cv": sc},
    open(f"{OUT}/cpu_meta_{OCR_TAG}.json", "w"),
)
sub = pd.DataFrame({"trace_id": test.trace_id, "root_code": [" ".join(x) for x in top3(P_te)]})
sub.to_csv(f"{OUT}/submission_cpu_{OCR_TAG}.csv", index=False)
print("wrote", f"{OUT}/submission_cpu_{OCR_TAG}.csv", sub.root_code.str.split().str[0].value_counts().head(5).to_dict())
