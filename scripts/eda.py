"""Quick look at the competition tables: shapes, label distribution, question overlap between the splits, and what the
question prior alone scores (leave-one-out MAP@3).

Usage:
    python scripts/eda.py
"""

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 110)
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
tax = pd.read_csv("data/taxonomy_v1.csv")
un = pd.read_csv("data/unlabelled.csv")
ss = pd.read_csv("data/sample_submission.csv")
print("shapes train/test/unlab/tax/ss", tr.shape, te.shape, un.shape, tax.shape, ss.shape)
print("train cols", list(tr.columns))
print("test cols", list(te.columns))
print("tax cols", list(tax.columns))
vc = tr.root_code.value_counts()
print("\nroot_code counts (top 30):")
print(vc.head(30).to_string())
print(
    "n classes in train:",
    vc.size,
    "| singletons:",
    (vc == 1).sum(),
    "| <=5:",
    (vc <= 5).sum(),
    "| codes in taxonomy absent from train:",
    len(set(tax.iloc[:, 0]) - set(vc.index)),
)
print(
    "\nNONE share:",
    (tr.root_code == "NONE").mean().round(3),
    "| final_answer_correct:",
    tr.final_answer_correct.mean().round(3),
    "| clean:",
    tr.clean.mean().round(3),
)
print("\ncrosstab root==NONE vs answer correct:")
print(pd.crosstab(tr.root_code == "NONE", tr.final_answer_correct))
print("\ngrade train:", tr.grade.value_counts().sort_index().to_dict())
print("grade test:", te.grade.value_counts().sort_index().to_dict())
print(
    "\nquestions: train",
    tr.question_id.nunique(),
    "test",
    te.question_id.nunique(),
    "test q in train:",
    te.question_id.isin(tr.question_id).mean().round(3),
    "unlab q:",
    un.question_id.nunique(),
    "unlab q in train:",
    un.question_id.isin(tr.question_id).mean().round(3),
)
print("rows/question train:", tr.question_id.value_counts().describe().round(1).to_dict())
g = tr.groupby("question_id").root_code.agg(lambda s: s.value_counts().iloc[0] / len(s))
print("mean majority-label share per question:", g.mean().round(3))


# question-prior MAP@3 estimate (leave-one-out)
def map3(preds, y):
    return sum((1 / (p.index(t) + 1)) if t in p else 0 for p, t in zip(preds, y)) / len(y)


glob = list(vc.index[:3])
preds = []
for i, r in tr.iterrows():
    sub = tr[(tr.question_id == r.question_id) & (tr.index != i)].root_code.value_counts()
    p = list(sub.index[:3])
    p += [c for c in glob if c not in p]
    preds.append(p[:3])
print(
    "LOO question-prior MAP@3:",
    round(map3(preds, tr.root_code.tolist()), 4),
    "| global-prior MAP@3:",
    round(map3([glob] * len(tr), tr.root_code.tolist()), 4),
)
print("\ntopic_name nunique:", tr.topic_name.nunique())
print(tr.topic_name.value_counts().head(8).to_string())
print(
    "\nq image share train:",
    tr.question_image_file.notna().mean().round(3),
    "test:",
    te.question_image_file.notna().mean().round(3),
)
print("\n--- taxonomy ---")
print(tax.to_string())
print("\n--- sample rows ---")
for i in [0, 3, 7]:
    r = tr.iloc[i]
    print("\n#", r.trace_id, r.root_code, "| grade", r.grade, "|", r.topic_name)
    print("Q:", str(r.question_text)[:300])
    print("KEY:", r.correct_answer)
    print("STEPS:", str(r.transcription_steps)[:400])
    print("MISC:", str(r.misconceptions)[:500])
print("\ntranscription chars:", tr.transcription_steps.str.len().describe().round(0).to_dict())
