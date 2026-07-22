import os
import math
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CSV_PATH        = "webnlg_qa_merged.csv"        # <-- your input
OUTPUT_PATH     = "webnlg_qa_merged_validated.csv"
BATCH_SIZE      = 32                     # tune to your VRAM
CHECKPOINT_EVERY = 100                    # save CSV every N rows
PRINT_EXAMPLE_EVERY = 1000                 # print a sample every N rows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ─────────────────────────────────────────────────────────────
# LOAD CSV — skip already-processed rows if restarting
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

if os.path.exists(OUTPUT_PATH):
    done_df = pd.read_csv(OUTPUT_PATH)
    start_idx = len(done_df)
    results = done_df.to_dict("records")
    print(f"Resuming from row {start_idx} / {len(df)} (found existing output)")
else:
    start_idx = 0
    results = []
    print(f"Starting fresh. Total rows: {len(df)}")

df_todo = df.iloc[start_idx:].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# LOAD BOOLQ MODELS
# ─────────────────────────────────────────────────────────────
DEBERTA_MODEL = "nfliu/deberta-v3-large_boolq"
ROBERTA_MODEL = "shahrukhx01/roberta-base-boolq"

print(f"\nLoading BoolQ model 1: {DEBERTA_MODEL}")
deberta_tok   = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL).to(device).eval()

print(f"Loading BoolQ model 2: {ROBERTA_MODEL}")
roberta_tok   = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device).eval()


def boolq_batch(model, tokenizer, pairs: list[tuple[str, str]], encoding_fn: str) -> list[tuple[str, float, float]]:
    """
    encoding_fn: 'deberta' | 'roberta'
    pairs: list of (question, context)
    Returns list of (label, p_yes, p_no)
    """
    if encoding_fn == "deberta":
        encoded = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
    else:  # roberta: encode_plus expects (question, passage) per sample
        encoded = tokenizer(
            [q for q, _ in pairs],
            [c for _, c in pairs],
            padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    out = []
    for p in probs:
        p_no, p_yes = round(p[0], 4), round(p[1], 4)
        label = "Yes" if p_yes >= p_no else "No"
        out.append((label, p_yes, p_no))
    return out


# ─────────────────────────────────────────────────────────────
# LOAD EXTRACTIVE QA MODELS
# ─────────────────────────────────────────────────────────────
EXTRACTIVE_MODELS = [
    "deepset/roberta-base-squad2",
    "deepset/deberta-v3-base-squad2",
]
print(f"\nLoading extractive QA models...")
qa_pipes = [
    pipeline(
        "question-answering", model=m, tokenizer=m,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=BATCH_SIZE
    )
    for m in EXTRACTIVE_MODELS
]


def extractive_batch(pipe, qa_inputs: list[dict]) -> list[tuple[str, float]]:
    """qa_inputs: list of {'question': ..., 'context': ...}"""
    results = pipe(qa_inputs, max_answer_len=50)
    if isinstance(results, dict):   # single item edge case
        results = [results]
    return [(r["answer"], round(r["score"], 4)) for r in results]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def soft_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a


def save_checkpoint(results: list, original_cols: list, eval_cols: list, path: str):
    out_df = pd.DataFrame(results)[original_cols + eval_cols]
    out_df.to_csv(path, index=False)


EVAL_COLS = [
    "boolq_deberta_pred", "boolq_deberta_p_yes", "boolq_deberta_p_no",
    "boolq_roberta_pred",  "boolq_roberta_p_yes",  "boolq_roberta_p_no",
    "qa_model1_pred",      "qa_model1_score",
    "qa_model2_pred",      "qa_model2_score",
    "corrected_answer",    "correction_flag",
]
ORIGINAL_COLS = list(df.columns)


# ─────────────────────────────────────────────────────────────
# SPLIT INTO BATCHES BY QUESTION TYPE
# ─────────────────────────────────────────────────────────────
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Collect indices by type
yesno_idxs = df_todo.index[df_todo["question_type"] == "yes_no"].tolist()
extr_idxs  = df_todo.index[df_todo["question_type"] == "extractive"].tolist()
other_idxs = df_todo.index[~df_todo["question_type"].isin(["yes_no", "extractive"])].tolist()

# Pre-allocate result slots (indexed by df_todo position)
result_slots = [None] * len(df_todo)


# ─────────────────────────────────────────────────────────────
# PROCESS YES/NO ROWS
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Processing {len(yesno_idxs)} yes/no rows (batch={BATCH_SIZE})")
print(f"{'─'*60}")

row_count = 0
for batch_idxs in tqdm(list(chunked(yesno_idxs, BATCH_SIZE)), desc="BoolQ batches", unit="batch"):
    rows   = df_todo.loc[batch_idxs]
    pairs  = [(r["question"], r["lex_text"]) for _, r in rows.iterrows()]

    d_preds = boolq_batch(deberta_model, deberta_tok,  pairs, "deberta")
    r_preds = boolq_batch(roberta_model, roberta_tok, pairs, "roberta")

    for i, (idx, (_, row)) in enumerate(zip(batch_idxs, rows.iterrows())):
        d_label, d_yes, d_no = d_preds[i]
        r_label, r_yes, r_no = r_preds[i]
        orig_ans = str(row["answer"]).strip()

        both_agree    = (d_label == r_label)
        matches_csv   = (d_label.lower() == orig_ans.lower())

        if both_agree and not matches_csv:
            corrected       = d_label
            correction_flag = f"CORRECTED: {orig_ans} → {corrected}"
        else:
            corrected       = orig_ans
            correction_flag = "MODELS_DISAGREE" if not both_agree else ""

        rec = row.to_dict()
        rec.update({
            "boolq_deberta_pred":  d_label, "boolq_deberta_p_yes": d_yes, "boolq_deberta_p_no": d_no,
            "boolq_roberta_pred":  r_label, "boolq_roberta_p_yes": r_yes, "boolq_roberta_p_no": r_no,
            "qa_model1_pred": "", "qa_model1_score": "",
            "qa_model2_pred": "", "qa_model2_score": "",
            "corrected_answer": corrected, "correction_flag": correction_flag,
        })
        result_slots[idx] = rec
        results.append(rec)
        row_count += 1

        # Print example
        if row_count % PRINT_EXAMPLE_EVERY == 0:
            tqdm.write(
                f"\n[Example row {start_idx + row_count}] yes/no\n"
                f"  Q:          {row['question']}\n"
                f"  Context:    {row['lex_text'][:80]}...\n"
                f"  CSV answer: {orig_ans}\n"
                f"  DeBERTa:    {d_label} (p_yes={d_yes})\n"
                f"  RoBERTa:    {r_label} (p_yes={r_yes})\n"
                f"  → {correction_flag or 'OK (no change)'}"
            )

        # Checkpoint
        if row_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(results, ORIGINAL_COLS, EVAL_COLS, OUTPUT_PATH)
            tqdm.write(f"  ✓ Checkpoint saved ({len(results)} rows)")


# ─────────────────────────────────────────────────────────────
# PROCESS EXTRACTIVE ROWS
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Processing {len(extr_idxs)} extractive rows (batch={BATCH_SIZE})")
print(f"{'─'*60}")

row_count = 0
for batch_idxs in tqdm(list(chunked(extr_idxs, BATCH_SIZE)), desc="Extractive batches", unit="batch"):
    rows      = df_todo.loc[batch_idxs]
    qa_inputs = [{"question": r["question"], "context": r["lex_text"]} for _, r in rows.iterrows()]

    preds1 = extractive_batch(qa_pipes[0], qa_inputs)
    preds2 = extractive_batch(qa_pipes[1], qa_inputs)

    for i, (idx, (_, row)) in enumerate(zip(batch_idxs, rows.iterrows())):
        pred1, score1 = preds1[i]
        pred2, score2 = preds2[i]
        orig_ans      = str(row["answer"]).strip()

        models_agree  = soft_match(pred1, pred2)
        matches_csv   = soft_match(pred1, orig_ans)

        if models_agree and not matches_csv:
            corrected       = pred1
            correction_flag = f"SUGGESTED: '{orig_ans}' → '{corrected}'"
        else:
            corrected       = orig_ans
            correction_flag = "MODELS_DISAGREE" if not models_agree else ""

        rec = row.to_dict()
        rec.update({
            "boolq_deberta_pred": "", "boolq_deberta_p_yes": "", "boolq_deberta_p_no": "",
            "boolq_roberta_pred": "", "boolq_roberta_p_yes": "", "boolq_roberta_p_no": "",
            "qa_model1_pred":  pred1, "qa_model1_score": score1,
            "qa_model2_pred":  pred2, "qa_model2_score": score2,
            "corrected_answer": corrected, "correction_flag": correction_flag,
        })
        result_slots[idx] = rec
        results.append(rec)
        row_count += 1

        if row_count % PRINT_EXAMPLE_EVERY == 0:
            tqdm.write(
                f"\n[Example row {start_idx + row_count}] extractive\n"
                f"  Q:          {row['question']}\n"
                f"  Context:    {row['lex_text'][:80]}...\n"
                f"  CSV answer: {orig_ans}\n"
                f"  QA model 1: '{pred1}' (score={score1})\n"
                f"  QA model 2: '{pred2}' (score={score2})\n"
                f"  → {correction_flag or 'OK (no change)'}"
            )

        if row_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(results, ORIGINAL_COLS, EVAL_COLS, OUTPUT_PATH)
            tqdm.write(f"  ✓ Checkpoint saved ({len(results)} rows)")


# ─────────────────────────────────────────────────────────────
# PROCESS OTHER / UNKNOWN ROWS
# ─────────────────────────────────────────────────────────────
for idx in other_idxs:
    row = df_todo.loc[idx]
    rec = row.to_dict()
    rec.update({k: "" for k in EVAL_COLS})
    rec["corrected_answer"] = str(row["answer"]).strip()
    rec["correction_flag"]  = "UNKNOWN_TYPE"
    results.append(rec)


# ─────────────────────────────────────────────────────────────
# FINAL SAVE
# ─────────────────────────────────────────────────────────────
save_checkpoint(results, ORIGINAL_COLS, EVAL_COLS, OUTPUT_PATH)
print(f"\n✅ Done. {len(results)} rows written to {OUTPUT_PATH}")