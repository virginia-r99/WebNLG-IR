import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import time
import csv
import sys
from typing import List, Dict, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen3-4B-Instruct-2507"
VALIDATED_CSV     = "webnlg_qa_dataset_validated.csv"   # output of validation step
COVERAGE_CSV      = "coverage_report.csv"        # output of lex-level coverage script
OUTPUT_CSV        = "webnlg_qa_dataset_rebuild.csv"      # original CSV — we append to this
BATCH_SIZE        = 8
PRINT_EVERY       = 20
SAVE_EVERY        = 30
N_GENERATE        = 3   # generate at least this many questions per missing type

CSV_FIELDNAMES = [
    "split", "category", "eid",
    "lex_id", "lex_text",
    "xml_file", "mtriple", "otriple",
    "statement",
    "question_idx", "question_type", "question", "answer",
    "generation_errors",
]

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS — one per missing type combination
# ─────────────────────────────────────────────────────────────

BASE_SYSTEM = """You are a data generation assistant for question answering.
Every question MUST be fully answerable using ONLY the information in the statement.
Do not assume, infer, or add outside knowledge. Output JSON only.
Schema: {"statement": "...", "questions": [{"question": "...", "answer": "...", "type": "yes_no" or "extractive"}]}"""

TYPE_INSTRUCTIONS = {
    "yes": f"Generate exactly {N_GENERATE} YES/NO questions whose answer is 'Yes'.",
    "no":  f"Generate exactly {N_GENERATE} YES/NO questions whose answer is 'No'. "
           f"Each question must be plausibly false given the statement (e.g. wrong country, wrong name).",
    "extractive": f"Generate exactly {N_GENERATE} extractive questions whose answer is a span from the statement.",
}

# ─────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES — one block per type
# ─────────────────────────────────────────────────────────────

FEW_SHOT = {
    "yes": [
        {
            "statement": "Arròs negre is a dish from Spain.",
            "output": {
                "statement": "Arròs negre is a dish from Spain.",
                "questions": [
                    {"question": "Is Arròs negre from Spain?",     "answer": "Yes", "type": "yes_no"},
                    {"question": "Is Arròs negre a Spanish dish?", "answer": "Yes", "type": "yes_no"},
                    {"question": "Does Arròs negre originate from Spain?", "answer": "Yes", "type": "yes_no"},
                ]
            }
        },
        {
            "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
            "output": {
                "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
                "questions": [
                    {"question": "Does Aarhus Airport serve Aarhus?",        "answer": "Yes", "type": "yes_no"},
                    {"question": "Is Aarhus Airport located in Denmark?",    "answer": "Yes", "type": "yes_no"},
                    {"question": "Is Aarhus Airport a Danish airport?",      "answer": "Yes", "type": "yes_no"},
                ]
            }
        }
    ],
    "no": [
        {
            "statement": "Arròs negre is a dish from Spain.",
            "output": {
                "statement": "Arròs negre is a dish from Spain.",
                "questions": [
                    {"question": "Is Arròs negre from France?",        "answer": "No", "type": "yes_no"},
                    {"question": "Is Arròs negre an Italian dish?",    "answer": "No", "type": "yes_no"},
                    {"question": "Does Arròs negre originate from Portugal?", "answer": "No", "type": "yes_no"},
                ]
            }
        },
        {
            "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
            "output": {
                "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
                "questions": [
                    {"question": "Does Aarhus Airport serve Copenhagen?",    "answer": "No", "type": "yes_no"},
                    {"question": "Is Aarhus Airport located in Sweden?",     "answer": "No", "type": "yes_no"},
                    {"question": "Does Aarhus Airport serve Oslo?",          "answer": "No", "type": "yes_no"},
                ]
            }
        }
    ],
    "extractive": [
        {
            "statement": "Arròs negre is a dish from Spain.",
            "output": {
                "statement": "Arròs negre is a dish from Spain.",
                "questions": [
                    {"question": "What is Arròs negre?",               "answer": "a dish", "type": "extractive"},
                    {"question": "Where is Arròs negre from?",         "answer": "Spain",  "type": "extractive"},
                    {"question": "Which country does Arròs negre come from?", "answer": "Spain", "type": "extractive"},
                ]
            }
        },
        {
            "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
            "output": {
                "statement": "The Aarhus Airport serves the city of Aarhus, Denmark.",
                "questions": [
                    {"question": "Which city does Aarhus Airport serve?",   "answer": "Aarhus",  "type": "extractive"},
                    {"question": "What airport serves Aarhus, Denmark?",    "answer": "The Aarhus Airport", "type": "extractive"},
                    {"question": "In which country is Aarhus Airport?",     "answer": "Denmark", "type": "extractive"},
                ]
            }
        }
    ]
}


def build_few_shot_text(qtype: str) -> str:
    lines = ["Here are examples of the format required:\n"]
    for ex in FEW_SHOT[qtype]:
        lines.append(f'Statement: {ex["statement"]}')
        lines.append(f'Output: {json.dumps(ex["output"], ensure_ascii=False)}\n')
    return "\n".join(lines)


def build_prompt(statement: str, missing_types: List[str], tokenizer) -> str:
    """Build a targeted prompt for exactly the missing types needed."""
    type_instructions = "\n".join(f"- {TYPE_INSTRUCTIONS[t]}" for t in missing_types)
    n_total = N_GENERATE * len(missing_types)

    # Use the first missing type's few-shot block (most critical missing type)
    few_shot_text = build_few_shot_text(missing_types[0])

    system = (
        BASE_SYSTEM + f"\n\nFor this task:\n{type_instructions}\n"
        f"Generate exactly {n_total} questions total. "
        f"Only the types listed above — do not generate other types."
    )

    user = (
        f"{few_shot_text}"
        f"Now generate for this statement.\n"
        f"Statement: {statement}\n"
        f"Return JSON only."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────────
# LOAD COVERAGE REPORT — identify what's missing per lex
# ─────────────────────────────────────────────────────────────

def load_missing_tasks(coverage_csv: str, validated_csv: str) -> List[Dict]:
    """
    Returns a list of tasks: one per (category, eid, lex_id) that is incomplete.
    Each task carries which types are missing and the original row metadata.
    """
    import pandas as pd

    cov = pd.read_csv(coverage_csv)
    val = pd.read_csv(validated_csv)

    # Only keep incomplete lexicalisations
    incomplete = cov[~cov["is_complete"]].copy()
    if len(incomplete) == 0:
        return []

    # Parse missing column into a list of missing types
    def parse_missing(missing_str: str) -> List[str]:
        types = []
        if pd.isna(missing_str) or missing_str == "":
            return types
        if "yes_question" in missing_str:
            types.append("yes")
        if "no_question" in missing_str:
            types.append("no")
        if "extractive_questions" in missing_str:
            types.append("extractive")
        return types

    incomplete["missing_types"] = incomplete["missing"].apply(parse_missing)

    # Attach metadata from validated CSV (split, xml_file, mtriple, otriple)
    meta_cols = ["category", "eid", "lex_id", "lex_text", "statement",
                 "split", "xml_file", "mtriple", "otriple"]
    meta_df = val[meta_cols].drop_duplicates(["category", "eid", "lex_id"])

    merged = incomplete.merge(meta_df, on=["category", "eid", "lex_id"], how="left")

    tasks = []
    for _, row in merged.iterrows():
        missing_types = row["missing_types"]
        if not missing_types:
            continue
        tasks.append({
            "split":         row.get("split", ""),
            "category":      row["category"],
            "eid":           row["eid"],
            "lex_id":        str(row["lex_id"]),
            "lex_text":      row.get("lex_text_x", row.get("lex_text", "")),
            "statement":     row.get("statement_x", row.get("statement", "")),
            "xml_file":      row.get("xml_file", ""),
            "mtriple":       row.get("mtriple", ""),
            "otriple":       row.get("otriple", ""),
            "missing_types": missing_types,
        })

    return tasks


# ─────────────────────────────────────────────────────────────
# RESUME — skip tasks already regenerated
# ─────────────────────────────────────────────────────────────

def load_regenerated_keys(output_csv: str) -> set:
    """
    Returns set of (eid, lex_id, question_type) already in the CSV
    so we don't re-generate a type that was regenerated in a previous run.
    """
    keys = set()
    if not os.path.exists(output_csv):
        return keys
    with open(output_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("generation_errors", "") == "" and row.get("question_type"):
                keys.add((row["eid"], row["lex_id"], row["question_type"]))
    return keys


# ─────────────────────────────────────────────────────────────
# JSON PARSE + VALIDATE
# ─────────────────────────────────────────────────────────────

def try_parse_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


TYPE_MAP = {
    "yes":        "yes_no",
    "no":         "yes_no",
    "extractive": "extractive",
}
ANSWER_MAP = {
    "yes": "Yes",
    "no":  "No",
}

def validate_and_filter(
    parsed: Optional[Dict],
    statement: str,
    missing_types: List[str],
) -> tuple[List[Dict], List[str]]:
    """
    Returns (filtered_questions, errors).
    Filters to only the missing types; validates polarity for yes/no.
    """
    errors = []
    if not parsed:
        return [], ["Invalid JSON"]
    if not isinstance(parsed.get("questions"), list):
        return [], ["Missing questions list"]

    wanted_csv_types = {TYPE_MAP[t] for t in missing_types}
    wanted_answers   = {ANSWER_MAP[t] for t in missing_types if t in ANSWER_MAP}

    filtered = []
    for qa in parsed["questions"]:
        qt  = qa.get("type", "")
        ans = str(qa.get("answer", "")).strip()

        if qt not in wanted_csv_types:
            continue  # skip unrequested types

        if qt == "yes_no":
            # Check answer polarity matches what we asked for
            if ans.lower() not in {a.lower() for a in wanted_answers}:
                continue  # wrong polarity — skip silently

        filtered.append(qa)

    if len(filtered) == 0:
        errors.append(f"No valid questions found for types {missing_types}")

    return filtered, errors


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────

def get_next_question_idx(output_csv: str, eid: str, lex_id: str) -> int:
    """Find the highest existing question_idx for this (eid, lex_id) and return next."""
    if not os.path.exists(output_csv):
        return 0
    max_idx = -1
    with open(output_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["eid"] == eid and row["lex_id"] == lex_id:
                try:
                    max_idx = max(max_idx, int(row["question_idx"]))
                except (ValueError, KeyError):
                    pass
    return max_idx + 1


def build_rows(task: Dict, questions: List[Dict], errors: List[str], start_idx: int) -> List[Dict]:
    base = {
        "split":    task["split"],
        "category": task["category"],
        "eid":      task["eid"],
        "lex_id":   task["lex_id"],
        "lex_text": task["lex_text"],
        "xml_file": task["xml_file"],
        "mtriple":  task["mtriple"],
        "otriple":  task["otriple"],
        "statement": task["statement"],
        "generation_errors": "; ".join(errors) if errors else "",
    }
    if not questions:
        base.update({"question_idx": "", "question_type": "", "question": "", "answer": ""})
        return [base]

    rows = []
    for i, qa in enumerate(questions):
        row = dict(base)
        row["question_idx"]  = start_idx + i
        row["question_type"] = qa.get("type", "unknown")
        row["question"]      = qa.get("question", "")
        row["answer"]        = qa.get("answer", "")
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────
# PRINT SAMPLE
# ─────────────────────────────────────────────────────────────

def print_sample(task: Dict, questions: List[Dict], errors: List[str],
                 global_idx: int, n_total: int, elapsed: float):
    status = "✅" if not errors else "❌"
    speed  = elapsed / max(global_idx, 1)
    eta    = speed * (n_total - global_idx)
    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"[{global_idx}/{n_total}] {status}  elapsed={elapsed:.0f}s  "
               f"speed={speed:.2f}s/it  ETA≈{eta:.0f}s")
    tqdm.write(f"  cat={task['category']}  eid={task['eid']}  lex_id={task['lex_id']}")
    tqdm.write(f"  missing was : {task['missing_types']}")
    tqdm.write(f"  statement   : {task['statement'][:100]}")
    if errors:
        tqdm.write(f"  ⚠ errors    : {errors}")
    for qa in questions:
        t = qa.get("type", "?")
        q = qa.get("question", "")
        a = qa.get("answer", "")
        tqdm.write(f"  [{t:10s}] Q: {q}")
        tqdm.write(f"             A: {a}")
    tqdm.write(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME):
    print(f"[LOG] Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    print(f"[LOG] Loading model: {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


# ─────────────────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_regeneration(
    tasks:      List[Dict],
    tokenizer,
    model,
    output_csv: str,
    batch_size: int = BATCH_SIZE,
    print_every: int = PRINT_EVERY,
    save_every:  int = SAVE_EVERY,
):
    n_total  = len(tasks)
    new_file = not os.path.exists(output_csv)
    csv_f    = open(output_csv, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
    if new_file:
        writer.writeheader()

    pending_rows = []
    n_ok    = 0
    n_skip  = 0
    start_t = time.time()

    # Pre-compute next question_idx for each (eid, lex_id) once
    idx_cache: Dict[tuple, int] = {}

    def get_start_idx(eid, lex_id):
        key = (eid, lex_id)
        if key not in idx_cache:
            idx_cache[key] = get_next_question_idx(output_csv, eid, lex_id)
        return idx_cache[key]

    def advance_idx(eid, lex_id, n):
        idx_cache[(eid, lex_id)] = idx_cache.get((eid, lex_id), 0) + n

    bar = tqdm(range(0, n_total, batch_size), desc="Regen batches", unit="batch")

    global_idx = 0

    try:
        for batch_start in bar:
            batch = tasks[batch_start : batch_start + batch_size]

            # NOTE: tasks in the same batch may have different missing types,
            # so prompts differ — that's fine, the model handles them independently.
            prompts = [
                build_prompt(t["statement"], t["missing_types"], tokenizer)
                for t in batch
            ]

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_len = inputs.input_ids.shape[1]
            decoded   = tokenizer.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )

            for task, raw_text in zip(batch, decoded):
                parsed = try_parse_json(raw_text)
                questions, errors = validate_and_filter(
                    parsed, task["statement"], task["missing_types"]
                )

                start_idx = get_start_idx(task["eid"], task["lex_id"])
                rows = build_rows(task, questions, errors, start_idx)
                advance_idx(task["eid"], task["lex_id"], len(questions))

                pending_rows.extend(rows)
                global_idx += 1

                if not errors:
                    n_ok += 1
                else:
                    n_skip += 1

                if global_idx % print_every == 0:
                    elapsed = time.time() - start_t
                    print_sample(task, questions, errors, global_idx, n_total, elapsed)

                if global_idx % save_every == 0:
                    for row in pending_rows:
                        writer.writerow(row)
                    csv_f.flush()
                    pending_rows.clear()
                    tqdm.write(
                        f"[SAVE] {global_idx}/{n_total} processed — "
                        f"ok={n_ok}  errors={n_skip}",
                        file=sys.stderr,
                    )

            elapsed = time.time() - start_t
            bar.set_postfix(ok=n_ok, err=n_skip, speed=f"{elapsed/max(global_idx,1):.2f}s/it")

    finally:
        for row in pending_rows:
            writer.writerow(row)
        csv_f.flush()
        csv_f.close()

    return n_ok, n_skip, global_idx


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[LOG] Loading coverage report and identifying missing tasks…", flush=True)
    tasks = load_missing_tasks(COVERAGE_CSV, VALIDATED_CSV)
    print(f"[LOG] Total incomplete lexicalisations: {len(tasks)}", flush=True)

    if not tasks:
        print("[LOG] Nothing to regenerate — all lexicalisations are complete.", flush=True)
        sys.exit(0)

    # ── Resume: filter out tasks where the missing type was already regenerated ──
    regen_keys = load_regenerated_keys(OUTPUT_CSV)
    if regen_keys:
        filtered = []
        for t in tasks:
            still_missing = [
                mt for mt in t["missing_types"]
                if (t["eid"], t["lex_id"], TYPE_MAP[mt]) not in regen_keys
            ]
            if still_missing:
                t = dict(t)
                t["missing_types"] = still_missing
                filtered.append(t)
        skipped = len(tasks) - len(filtered)
        print(f"[LOG] Resume: skipping {skipped} already-regenerated tasks. "
              f"Remaining: {len(filtered)}", flush=True)
        tasks = filtered

    if not tasks:
        print("[LOG] Nothing left to regenerate.", flush=True)
        sys.exit(0)

    # ── Breakdown of what's missing ──
    from collections import Counter
    type_counter = Counter(mt for t in tasks for mt in t["missing_types"])
    print(f"[LOG] Missing type breakdown: {dict(type_counter)}", flush=True)

    tokenizer, model = load_model()

    start = time.time()
    n_ok, n_err, n_total = run_regeneration(
        tasks, tokenizer, model,
        output_csv=OUTPUT_CSV,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
    )
    duration = time.time() - start

    print("\n" + "=" * 80, flush=True)
    print("REGENERATION COMPLETE", flush=True)
    print(f"  Processed : {n_total} tasks in {duration:.1f}s ({duration/max(n_total,1):.2f}s/task)", flush=True)
    print(f"  Success   : {n_ok}/{n_total}  |  Errors: {n_err}", flush=True)
    print(f"  Output    : {OUTPUT_CSV}  (new rows appended)", flush=True)
    print("=" * 80, flush=True)