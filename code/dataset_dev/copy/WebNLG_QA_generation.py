import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import time
import csv
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

WEBNLG_ROOT = "./WebNLG_ES"       # adjust to your dataset folder
OUTPUT_CSV  = "webnlg_qa_dataset.csv"
BATCH_SIZE  = 8
PRINT_EVERY = 25   # print a sample every N statements
SAVE_EVERY  = 50   # flush rows to CSV every N statements

SYSTEM_PROMPT = """You are a data generation assistant for question answering.
Your task is to generate natural, answerable questions that can be directly answered from a factual statement. Every question MUST be fully answerable using ONLY the information in the statement — do not assume, infer, or add outside knowledge.
Generate both yes/no questions AND plain extractive QA questions.
Output JSON only. Generate exactly 6 questions: 2 yes/no questions with answer "Yes", 2 yes/no questions with answer "No", and 2 extractive QA questions (total 6 questions). 
Schema: {"statement": "...", "questions": [{"question": "...", "answer": "...", "type": "yes_no" or "extractive"}]}"""

FEW_SHOT_EXAMPLES = [
    {
        "statement": "Arròs negre is from Spain.",
        "output": {
            "statement": "Arròs negre is from Spain.",
            "questions": [
                {"question": "Is Arròs negre from Spain?",       "answer": "Yes",   "type": "yes_no"},
                {"question": "Is Arròs negre a Spanish dish?",   "answer": "Yes",   "type": "yes_no"},
                {"question": "Is Arròs negre from France?",      "answer": "No",    "type": "yes_no"},
                {"question": "Is Arròs negre from Italy?",       "answer": "No",    "type": "yes_no"},
                {"question": "Which country is Arròs negre from?",     "answer": "Spain", "type": "extractive"},
                {"question": "Where does Arròs negre originate from?", "answer": "Spain", "type": "extractive"},
            ]
        }
    }
]

CSV_FIELDNAMES = [
    "split", "category", "eid",
    "lex_id", "lex_text",          # lex index + full text of the lexicalisation
    "xml_file", "mtriple", "otriple",
    "statement",
    "question_idx", "question_type", "question", "answer",
    "generation_errors",
]


def log(msg: str):
    print(f"[LOG] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# WebNLG Loading
# ──────────────────────────────────────────────────────────────────────────────

def parse_webnlg_xml(xml_path: str, split: str) -> List[Dict]:
    entries = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        log(f"  XML parse error in {xml_path}: {e}")
        return entries

    for entry in root.iter("entry"):
        if entry.get("size") != "1":
            continue

        category = entry.get("category", "")
        eid      = entry.get("eid", "")

        lex_texts = [
            lex.text.strip()
            for lex in entry.findall("lex")
            if lex.get("lang") == "en" and lex.text
        ]
        mtriples = [mt.text.strip() for mt in entry.iter("mtriple") if mt.text]
        otriples = [ot.text.strip() for ot in entry.iter("otriple") if ot.text]

        if not lex_texts:
            continue

        entries.append({
            "split":      split,
            "category":   category,
            "eid":        eid,
            "xml_file":   os.path.basename(xml_path),
            "statements": lex_texts,
            "mtriples":   mtriples,
            "otriples":   otriples,
        })

    return entries


def load_webnlg_size1(root_dir: str) -> List[Dict]:
    all_entries = []

    for split in ("train", "dev"):
        split_dir = os.path.join(root_dir, split, "1triples")
        if not os.path.isdir(split_dir):
            candidates = glob.glob(os.path.join(root_dir, split, "*triples*"))
            split_dir = candidates[0] if candidates else None

        if split_dir and os.path.isdir(split_dir):
            xml_files = glob.glob(os.path.join(split_dir, "*.xml"))
            log(f"  {split}/1triples — found {len(xml_files)} XML file(s)")
            for xf in sorted(xml_files):
                all_entries.extend(parse_webnlg_xml(xf, split))
        else:
            log(f"  WARNING: could not find 1triples folder for split='{split}'")

    test_dir = os.path.join(root_dir, "test")
    if os.path.isdir(test_dir):
        xml_files = glob.glob(os.path.join(test_dir, "*.xml"))
        log(f"  test/ — found {len(xml_files)} XML file(s)")
        for xf in sorted(xml_files):
            all_entries.extend(parse_webnlg_xml(xf, "test"))
    else:
        log(f"  WARNING: test/ folder not found")

    log(f"Total size-1 entries loaded: {len(all_entries)}")
    return all_entries


# ──────────────────────────────────────────────────────────────────────────────
# Resume helpers — detect already-processed (eid, lex_id) pairs
# ──────────────────────────────────────────────────────────────────────────────

def load_processed_keys(output_path: str) -> set:
    """Return a set of (eid, lex_id) tuples already present in the CSV."""
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add((row["eid"], row["lex_id"]))
    log(f"  Resume: {len(processed)} (eid, lex_id) pairs already in CSV — skipping.")
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME):
    log(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    log(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def build_prompt(statement: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Statement: {statement}\nReturn JSON only."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def validate_output(parsed: Optional[Dict], statement: str) -> List[str]:
    if not parsed:
        return ["Invalid JSON"]
    errors = []
    if parsed.get("statement") != statement:
        errors.append("Statement mismatch")
    if not isinstance(parsed.get("questions"), list) or len(parsed["questions"]) < 1:
        errors.append("Missing questions")
    return errors


# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_csv_writer(output_path: str, write_header: bool):
    """Open CSV in append mode and return (file_handle, DictWriter)."""
    f = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if write_header:
        writer.writeheader()
    return f, writer


def flush_rows(writer, csv_file, pending_rows: list):
    """Write pending rows to CSV and flush to disk."""
    for row in pending_rows:
        writer.writerow(row)
    csv_file.flush()
    pending_rows.clear()


def build_rows(meta: Dict, res: Dict) -> List[Dict]:
    """Convert one (metadata, generation result) pair into CSV row(s)."""
    base = {
        "split":    meta["split"],
        "category": meta["category"],
        "eid":      meta["eid"],
        "lex_id":   meta["lex_id"],
        "lex_text": meta["lex_text"],
        "xml_file": meta["xml_file"],
        "mtriple":  " | ".join(meta["mtriples"]),
        "otriple":  " | ".join(meta["otriples"]),
        "statement": res["statement"],
        "generation_errors": "; ".join(res["errors"]) if res["errors"] else "",
    }

    parsed = res.get("parsed")
    if parsed and isinstance(parsed.get("questions"), list):
        rows = []
        for q_idx, qa in enumerate(parsed["questions"]):
            row = dict(base)
            row["question_idx"]  = q_idx
            row["question_type"] = qa.get("type", "unknown")
            row["question"]      = qa.get("question", "")
            row["answer"]        = qa.get("answer", "")
            rows.append(row)
        return rows
    else:
        base.update({"question_idx": "", "question_type": "", "question": "", "answer": ""})
        return [base]


# ──────────────────────────────────────────────────────────────────────────────
# Print sample helper
# ──────────────────────────────────────────────────────────────────────────────

def print_sample(res: Dict, meta: Dict, global_idx: int, n_total: int, elapsed: float):
    status = "✅" if not res["errors"] else "❌"
    speed  = elapsed / max(global_idx, 1)
    eta    = speed * (n_total - global_idx)
    print(f"\n{'─'*60}", flush=True)
    print(f"[{global_idx}/{n_total}] {status}  elapsed={elapsed:.0f}s  "
          f"speed={speed:.2f}s/item  ETA≈{eta:.0f}s", flush=True)
    print(f"  split={meta['split']}  cat={meta['category']}  "
          f"eid={meta['eid']}  lex_id={meta['lex_id']}", flush=True)
    print(f"  LEX TEXT : {meta['lex_text']}", flush=True)
    if res["errors"]:
        print(f"  ⚠ errors : {', '.join(res['errors'])}", flush=True)
    parsed = res.get("parsed")
    if parsed and "questions" in parsed:
        for i, qa in enumerate(parsed["questions"], 1):
            t = qa.get("type", "?")
            q = qa.get("question", "N/A")
            a = qa.get("answer", "N/A")
            print(f"  {i}. [{t:10s}] Q: {q}", flush=True)
            print(f"              A: {a}", flush=True)
    else:
        print(f"  RAW snippet: {res['raw_output'][:120]}…", flush=True)
    print(f"{'─'*60}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop  (streaming — no giant in-memory list)
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_generation(
    flat_entries:    List[Dict],
    flat_statements: List[str],
    tokenizer,
    model,
    output_path:     str,
    batch_size:      int = BATCH_SIZE,
    print_every:     int = PRINT_EVERY,
    save_every:      int = SAVE_EVERY,
):
    n_total  = len(flat_statements)
    new_file = not os.path.exists(output_path)
    csv_file, writer = open_csv_writer(output_path, write_header=new_file)

    pending_rows: List[Dict] = []
    n_ok = 0
    start_time = time.time()

    batch_bar = tqdm(
        range(0, n_total, batch_size),
        desc="Batches",
        unit="batch",
        dynamic_ncols=True,
    )

    global_idx = 0   # statement counter (0-based)

    try:
        for batch_start in batch_bar:
            batch_meta  = flat_entries[batch_start : batch_start + batch_size]
            batch_stmts = flat_statements[batch_start : batch_start + batch_size]

            prompts = [build_prompt(s, tokenizer) for s in batch_stmts]
            inputs  = tokenizer(
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

            input_len       = inputs.input_ids.shape[1]
            decoded_outputs = tokenizer.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )

            for meta, statement, raw_text in zip(batch_meta, batch_stmts, decoded_outputs):
                parsed = try_parse_json(raw_text)
                errors = validate_output(parsed, statement)
                res = {
                    "statement":  statement,
                    "raw_output": raw_text,
                    "parsed":     parsed,
                    "errors":     errors,
                }
                if not errors:
                    n_ok += 1

                pending_rows.extend(build_rows(meta, res))
                global_idx += 1

                # periodic print
                if global_idx % print_every == 0:
                    elapsed = time.time() - start_time
                    print_sample(res, meta, global_idx, n_total, elapsed)

                # periodic CSV flush
                if global_idx % save_every == 0:
                    flush_rows(writer, csv_file, pending_rows)
                    tqdm.write(
                        f"[SAVE] {global_idx}/{n_total} statements processed — "
                        f"CSV flushed ({n_ok} ok so far)",
                        file=__import__("sys").stderr,
                    )

            # update batch progress bar postfix
            elapsed = time.time() - start_time
            batch_bar.set_postfix(
                ok=n_ok,
                done=global_idx,
                speed=f"{elapsed/max(global_idx,1):.2f}s/it",
            )

    finally:
        # Always flush remaining rows and close the file
        if pending_rows:
            flush_rows(writer, csv_file, pending_rows)
        csv_file.close()

    return n_ok, global_idx


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer, model = load_model()

    # ── 1. Load WebNLG size-1 entries ──────────────────────────────────────
    log("Loading WebNLG size-1 entries …")
    webnlg_entries = load_webnlg_size1(WEBNLG_ROOT)

    if not webnlg_entries:
        raise RuntimeError(
            f"No size-1 entries found. Check WEBNLG_ROOT='{WEBNLG_ROOT}' "
            "(should contain train/, dev/, test/ folders)."
        )

    # ── 2. Flatten: one task per (entry × lex_text) ────────────────────────
    flat_entries: List[Dict] = []
    flat_statements: List[str] = []

    for entry in webnlg_entries:
        for lex_idx, stmt in enumerate(entry["statements"]):
            meta = dict(entry)
            meta["lex_id"]   = str(lex_idx)
            meta["lex_text"] = stmt          # full lexicalisation text
            flat_entries.append(meta)
            flat_statements.append(stmt)

    log(f"Total statements to process: {len(flat_statements)}")

    # ── 3. Resume: skip already-done (eid, lex_id) pairs ───────────────────
    processed_keys = load_processed_keys(OUTPUT_CSV)
    if processed_keys:
        filtered_entries    = []
        filtered_statements = []
        for meta, stmt in zip(flat_entries, flat_statements):
            if (meta["eid"], meta["lex_id"]) not in processed_keys:
                filtered_entries.append(meta)
                filtered_statements.append(stmt)
        skipped = len(flat_entries) - len(filtered_entries)
        log(f"Skipping {skipped} already-processed statements. "
            f"Remaining: {len(filtered_entries)}")
        flat_entries    = filtered_entries
        flat_statements = filtered_statements

    if not flat_statements:
        log("Nothing left to process — CSV is already complete.")
        exit(0)

    # ── 4. Generate + stream-write CSV ─────────────────────────────────────
    log(f"Starting generation (batch_size={BATCH_SIZE}, "
        f"print_every={PRINT_EVERY}, save_every={SAVE_EVERY}) …")
    start = time.time()

    n_ok, n_total = run_generation(
        flat_entries, flat_statements, tokenizer, model,
        output_path=OUTPUT_CSV,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
    )

    duration = time.time() - start
    print("\n" + "=" * 80, flush=True)
    print("GENERATION COMPLETE", flush=True)
    print(f"  Processed : {n_total} statements in {duration:.1f}s "
          f"({duration/max(n_total,1):.2f}s/item)", flush=True)
    print(f"  Success   : {n_ok}/{n_total}", flush=True)
    print(f"  Output    : {OUTPUT_CSV}", flush=True)
    print("=" * 80, flush=True)