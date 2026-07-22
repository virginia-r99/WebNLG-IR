import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import csv
import json
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_CSV   = "webnlg_qa_selected_es_v3.csv"
OUTPUT_CSV  = "webnlg_qa_validated_es.csv"

BATCH_SIZE  = 16    # LLM batch — tune to VRAM
PRINT_EVERY = 100
SAVE_EVERY  = 200

LLM_MODEL   = "Qwen/Qwen3-4B"

YES_EQUIVALENTS = {"yes", "sí", "si"}
NO_EQUIVALENTS  = {"no"}


def log(msg):
    print(f"[LOG] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# YES/NO ANSWER  (pure rule, no LLM needed)
# ─────────────────────────────────────────────────────────────

def check_yesno(en: str, es: str) -> tuple[bool, str]:
    en_l, es_l = en.strip().lower(), es.strip().lower()
    en_yes = en_l in YES_EQUIVALENTS
    en_no  = en_l in NO_EQUIVALENTS
    es_yes = es_l in YES_EQUIVALENTS
    es_no  = es_l in NO_EQUIVALENTS
    if not (en_yes or en_no):
        return False, f"Unexpected EN: '{en}'"
    if not (es_yes or es_no):
        return False, f"Unexpected ES: '{es}'"
    if (en_yes and es_yes) or (en_no and es_no):
        return True, f"{'Yes↔Sí' if en_yes else 'No↔No'} [rule]"
    return False, f"Mismatch EN='{en}' ES='{es}' [rule]"


# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a bilingual English-Spanish equivalence checker.
You will receive an English text and a Spanish text.
They may be questions or short answers.
Decide if they are equivalent in meaning — allow for:
  - natural translation differences
  - different but synonymous words (e.g. "ground" / "estadio")
  - short span vs full sentence containing that span
  - translated proper names (e.g. "Spain" / "España")
  - yes/no in either language

Reply with JSON only, no explanation outside JSON:
{"ok": true} or {"ok": false}"""

FEW_SHOT = [
    (
        'English: "What is the ground used by A.S. Livorno Calcio?"\n'
        'Spanish: "¿Cuál es el nombre del estadio utilizado por el A.S. Livorno Calcio?"',
        '{"ok": true}'
    ),
    (
        'English: "AEK Athens FC"\n'
        'Spanish: "El AEK Atenas FC jugó en la temporada 2014"',
        '{"ok": true}'
    ),
    (
        'English: "Is the capital of France Paris?"\n'
        'Spanish: "¿Cuál es la capital de Francia?"',
        '{"ok": false}'
    ),
    (
        'English: "Yes"\n'
        'Spanish: "Sí"',
        '{"ok": true}'
    ),
    (
        'English: "No"\n'
        'Spanish: "No"',
        '{"ok": true}'
    ),
    (
        'English: "Italy"\n'
        'Spanish: "España"',
        '{"ok": false}'
    ),
]


# ── 2. Add /no_think to the user turn in build_prompt ────────
def build_prompt(en: str, es: str, tokenizer) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEW_SHOT:
        messages.append({"role": "user",      "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({
        "role": "user",
        "content": f'English: "{en}"\nSpanish: "{es}" /no_think'  # suppress CoT
    })
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_llm():
    log(f"Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def parse_ok(text: str) -> bool | None:
    """Extract the 'ok' boolean from model output. Returns None if unparseable."""
    # strip thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return bool(json.loads(text)["ok"])
    except Exception:
        m = re.search(r'"ok"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        # last resort: look for bare true/false
        if re.search(r'\btrue\b', text, re.IGNORECASE):
            return True
        if re.search(r'\bfalse\b', text, re.IGNORECASE):
            return False
        return None


@torch.inference_mode()
def llm_judge_batch(
    pairs: List[tuple[str, str]],   # list of (en, es)
    tokenizer,
    model,
    max_new_tokens: int = 64,
) -> List[tuple[bool, str]]:
    """
    Returns list of (ok: bool, raw_output: str) for each pair.
    Unparseable outputs default to False.
    """
    prompts = [build_prompt(en, es, tokenizer) for en, es in pairs]
    inputs  = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True,
    ).to(model.device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    decoded   = tokenizer.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )

    results = []
    for raw in decoded:
        ok = parse_ok(raw)
        if ok is None:
            ok = False   # conservative fallback
        results.append((ok, raw.strip()))
    return results


# ─────────────────────────────────────────────────────────────
# CSV / RESUME
# ─────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "split", "category", "eid", "lex_id",
    "lex_text", "lex_text_es",
    "xml_file", "mtriple", "otriple", "statement",
    "question_idx", "question_type",
    "question", "answer",
    "question_es", "answer_es",
    "generation_errors", "corrected_answer", "correction_flag",
    "selected_role", "selection_source",
    # ── validation ───────────────────────────────────────────
    "val_question_ok",
    "val_question_llm_raw",
    "val_question_reason",
    "val_answer_method",      # "rule" | "llm"
    "val_answer_ok",
    "val_answer_llm_raw",
    "val_answer_reason",
    "val_overall_ok",
]


def load_processed_keys(path: str) -> set:
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            keys.add((r["eid"], r["lex_id"], r["question_idx"]))
    log(f"  Resume: {len(keys)} rows already done.")
    return keys


def open_writer(path: str, write_header: bool):
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    if write_header:
        w.writeheader()
    return f, w


def flush(writer, f, rows: list):
    for r in rows:
        writer.writerow(r)
    f.flush()
    rows.clear()
    pd.read_csv(OUTPUT_CSV).to_excel(
        OUTPUT_CSV.replace(".csv", ".xlsx"), index=False, engine="openpyxl"
    )


# ─────────────────────────────────────────────────────────────
# PRINT SAMPLE
# ─────────────────────────────────────────────────────────────

def print_sample(row: Dict, idx: int, n: int, elapsed: float):
    overall = "✅" if row["val_overall_ok"] == "True" else "❌"
    speed   = elapsed / max(idx, 1)
    print(f"\n{'─'*65}", flush=True)
    print(f"[{idx}/{n}] {overall}  elapsed={elapsed:.0f}s  "
          f"speed={speed:.2f}s/it  ETA≈{speed*(n-idx):.0f}s", flush=True)
    print(f"  {row['split']} | {row['category']} | eid={row['eid']} "
          f"lex={row['lex_id']} q={row['question_idx']} [{row['question_type']}]",
          flush=True)
    print(f"  EN Q : {row['question']}", flush=True)
    print(f"  ES Q : {row['question_es']}", flush=True)
    print(f"  EN A : {row['answer']}", flush=True)
    print(f"  ES A : {row['answer_es']}", flush=True)
    q_icon = "✅" if row["val_question_ok"] == "True" else "❌"
    a_icon = "✅" if row["val_answer_ok"]   == "True" else "❌"
    print(f"  {q_icon} question [{row['val_question_reason']}]", flush=True)
    print(f"  {a_icon} answer   [{row['val_answer_reason']}]", flush=True)
    print(f"{'─'*65}", flush=True)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def run_validation(rows, tokenizer, model, output_path,
                   batch_size=BATCH_SIZE, print_every=PRINT_EVERY, save_every=SAVE_EVERY):
    n_total  = len(rows)
    new_file = not os.path.exists(output_path)
    csv_f, writer = open_writer(output_path, write_header=new_file)

    pending = []
    n_ok    = 0
    t0      = time.time()
    g_idx   = 0

    bar = tqdm(range(0, n_total, batch_size), desc="Validating",
               unit="batch", dynamic_ncols=True)

    try:
        for bs in bar:
            batch     = rows[bs : bs + batch_size]
            extr_mask = [r["question_type"] == "extractive" for r in batch]
            extr_idxs = [i for i, m in enumerate(extr_mask) if m]

            # ── LLM call 1: all questions ─────────────────────
            q_pairs   = [(r["question"], r["question_es"]) for r in batch]
            q_results = llm_judge_batch(q_pairs, tokenizer, model)

            # ── LLM call 2: extractive answers only ───────────
            a_pairs_extr  = [(batch[i]["answer"], batch[i]["answer_es"]) for i in extr_idxs]
            a_results_extr = llm_judge_batch(a_pairs_extr, tokenizer, model) if a_pairs_extr else []
            a_llm_map      = dict(zip(extr_idxs, a_results_extr))

            # ── Assemble rows ─────────────────────────────────
            for i, row in enumerate(batch):
                out = dict(row)

                q_ok, q_raw = q_results[i]
                q_reason    = f"LLM={'ok' if q_ok else 'fail'} raw={q_raw!r}"

                if not extr_mask[i]:
                    a_ok, a_reason = check_yesno(row["answer"], row["answer_es"])
                    a_raw    = ""
                    a_method = "rule"
                else:
                    a_ok, a_raw = a_llm_map[i]
                    a_reason    = f"LLM={'ok' if a_ok else 'fail'} raw={a_raw!r}"
                    a_method    = "llm"

                overall = q_ok and a_ok
                if overall:
                    n_ok += 1

                out.update({
                    "val_question_ok":      str(q_ok),
                    "val_question_llm_raw": q_raw,
                    "val_question_reason":  q_reason,
                    "val_answer_method":    a_method,
                    "val_answer_ok":        str(a_ok),
                    "val_answer_llm_raw":   a_raw,
                    "val_answer_reason":    a_reason,
                    "val_overall_ok":       str(overall),
                })

                pending.append(out)
                g_idx += 1

                if g_idx % print_every == 0:
                    print_sample(out, g_idx, n_total, time.time() - t0)

                if g_idx % save_every == 0:
                    flush(writer, csv_f, pending)
                    tqdm.write(f"[SAVE] {g_idx}/{n_total} — {n_ok} ok",
                               file=__import__("sys").stderr)

            bar.set_postfix(ok=n_ok, done=g_idx,
                            speed=f"{(time.time()-t0)/max(g_idx,1):.2f}s/it")

    finally:
        if pending:
            flush(writer, csv_f, pending)
        csv_f.close()

    return n_ok, g_idx


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer, model = load_llm()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    log(f"Total rows: {len(all_rows)}")

    todo = [r for r in all_rows
            if r.get("question_es", "").strip() and r.get("answer_es", "").strip()]
    log(f"Skipped (no ES): {len(all_rows) - len(todo)}")

    done = load_processed_keys(OUTPUT_CSV)
    if done:
        todo = [r for r in todo
                if (r["eid"], r["lex_id"], r["question_idx"]) not in done]
    log(f"To validate: {len(todo)}")

    if not todo:
        log("Nothing to do.")
        exit(0)

    t0 = time.time()
    n_ok, n_total = run_validation(todo, tokenizer, model, OUTPUT_CSV)
    dur = time.time() - t0

    df    = pd.read_csv(OUTPUT_CSV)
    yesno = df[df["question_type"] == "yes_no"]
    extr  = df[df["question_type"] == "extractive"]

    def pct(a, b): return f"{100*a/max(b,1):.1f}%"

    print("\n" + "=" * 80, flush=True)
    print("VALIDATION COMPLETE", flush=True)
    print(f"  Rows       : {n_total} in {dur:.1f}s ({dur/max(n_total,1):.2f}s/row)", flush=True)
    print(f"  Overall ✅  : {n_ok}/{n_total} ({pct(n_ok, n_total)})", flush=True)
    nq = (df["val_question_ok"] == "True").sum()
    print(f"\n  Question [LLM]  pass: {nq}/{len(df)} ({pct(nq, len(df))})", flush=True)
    ny = (yesno["val_answer_ok"] == "True").sum()
    print(f"  Answer yes/no [rule] pass: {ny}/{len(yesno)} ({pct(ny, len(yesno))})", flush=True)
    na = (extr["val_answer_ok"] == "True").sum()
    print(f"  Answer extractive [LLM] pass: {na}/{len(extr)} ({pct(na, len(extr))})", flush=True)

    xlsx_path = OUTPUT_CSV.replace(".csv", ".xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    log(f"Excel saved: {xlsx_path}")
    print(f"\n  Output CSV   : {OUTPUT_CSV}", flush=True)
    print(f"  Output Excel : {xlsx_path}", flush=True)
    print("=" * 80, flush=True)