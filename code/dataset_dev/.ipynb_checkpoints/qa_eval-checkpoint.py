import pandas as pd
from transformers import pipeline
from io import StringIO
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # swap for any NLI HF model
# e.g. "facebook/bart-large-mnli", "typeform/distilbart-mnli-12-3" (lighter)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
CSV_PATH = "webnlg_qa_dataset.csv"   # <-- replace with your path
df = pd.read_csv(CSV_PATH)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
nli = pipeline(
    "zero-shot-classification",          # uses NLI under the hood
    model=NLI_MODEL,
    device=0,                            # set to -1 for CPU
)

# Alternatively, use the raw NLI pipeline for finer control:
# from transformers import pipeline
# nli = pipeline("text-classification", model=NLI_MODEL, device=0)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def nli_score(premise: str, hypothesis: str) -> dict:
    """
    Returns entailment/neutral/contradiction probabilities.
    Uses zero-shot-classification with a single candidate label trick.
    """
    result = nli(
        sequences=premise,
        candidate_labels=["entailment", "contradiction", "neutral"],
        hypothesis_template="{}",
        # DeBERTa NLI models expect: premise=sequences, hypothesis=candidate_labels fmt
        # For raw NLI pipeline use cross_encoder style (see note below)
    )
    label_scores = dict(zip(result["labels"], result["scores"]))
    return label_scores

def check_faithfulness(statement: str, answer: str) -> tuple[str, float]:
    """
    Check: does the statement entail the answer?
    HIGH entailment → answer is grounded in the statement ✓
    """
    scores = nli_score(premise=statement, hypothesis=answer)
    label = max(scores, key=scores.get)
    return label, round(scores.get("entailment", 0), 4)

def check_alignment(statement: str, question: str, answer: str) -> tuple[str, float]:
    """
    Check: does the statement contain info to answer the question?
    We treat the statement as premise and (question + answer) as hypothesis.
    """
    hypothesis = f"The answer to '{question}' is '{answer}'."
    scores = nli_score(premise=statement, hypothesis=hypothesis)
    label = max(scores, key=scores.get)
    return label, round(scores.get("entailment", 0), 4)

# ── EVALUATE ─────────────────────────────────────────────────────────────────
faithfulness_labels, faithfulness_scores = [], []
alignment_labels, alignment_scores = [], []

for _, row in tqdm(df.iterrows()):
    statement = str(row["lex_text"])  # "statement" in your data is lex_text
    question  = str(row["question"])
    answer    = str(row["answer"])

    fl, fs = check_faithfulness(statement, answer)
    al, as_ = check_alignment(statement, question, answer)

    faithfulness_labels.append(fl)
    faithfulness_scores.append(fs)
    alignment_labels.append(al)
    alignment_scores.append(as_)

df["faithfulness_label"]  = faithfulness_labels   # entailment/neutral/contradiction
df["faithfulness_score"]  = faithfulness_scores   # P(entailment): 0→1
df["alignment_label"]     = alignment_labels
df["alignment_score"]     = alignment_scores

# ── VERDICT ──────────────────────────────────────────────────────────────────
# A row passes if BOTH checks return entailment
df["verdict"] = df.apply(
    lambda r: "✅ PASS" if r["faithfulness_label"] == "entailment"
                        and r["alignment_label"]   == "entailment"
              else "❌ FAIL",
    axis=1
)

# ── OUTPUT ───────────────────────────────────────────────────────────────────
output_cols = [
    "split", "eid", "lex_id", "question_idx", "question_type",
    "lex_text", "question", "answer",
    "faithfulness_label", "faithfulness_score",
    "alignment_label",    "alignment_score",
    "verdict"
]
df[output_cols].to_csv("qa_evaluation_results.csv", index=False)
print(df[output_cols].to_string())