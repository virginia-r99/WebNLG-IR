import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import time
import csv
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
LLM_MODEL        = "Qwen/Qwen3-4B"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"   # EN→ES MarianMT
WEBNLG_ROOT      = "./WebNLG_ES"
INPUT_CSV        = "webnlg_qa_selected.csv"
OUTPUT_CSV       = "webnlg_qa_selected_es_v3.csv"

BATCH_SIZE   = 8
PRINT_EVERY  = 25
SAVE_EVERY   = 50
MAX_RETRIES  = 3

YES_EQUIVALENTS = {"yes", "sí", "si"}
NO_EQUIVALENTS  = {"no"}

# ─────────────────────────────────────────────────────────────
# STEP 2 — LLM ENTITY CORRECTOR
# ─────────────────────────────────────────────────────────────

CORRECTOR_SYSTEM = """\
You are a post-editor for Spanish QA data in español de España.
You will receive:
- An English question (original meaning — ground truth)
- A Spanish translation of that question (produced by a machine translator)
- A Spanish lexicalisation of the related fact (reference for entity names and style)
- The English answer (to help identify the entity being asked about)

Your task: correct ONLY what is necessary to make the Spanish question coherent with the lexicalisation.

Rules:
1. ENTITY NAMES ONLY: if the translation uses a Spanish name for an entity but the lexicalisation
   uses a different form (e.g. English name, or different spelling), replace it with the lexicalisation form.
   Example: translation says "La Alhambra fue terminada" → lexicalisation says "completada" → BOTH are fine, do NOT change.
   Example: translation says "el aeropuerto de Ámsterdam" → lexicalisation says "aeropuerto de Schiphol" → fix to match.
2. DO NOT change correct synonyms, verb tenses, or grammatical structures that preserve the meaning.
3. DO NOT add or remove information from the question.
4. The question must stay a well-formed Spanish interrogative (¿...?).
5. If the translation is already coherent with the lexicalisation, return it EXACTLY as-is, with no changes at all.

Output JSON only. Schema:
{"question_es": "..."}"""

CORRECTOR_FEW_SHOT = [
    # 1. Entity name mismatch: airport name
    (
        (
            "English question   : What is the runway length of Amsterdam Airport Schiphol in metres?\n"
            "Spanish translation: ¿Cuál es la longitud de la pista del aeropuerto de Ámsterdam en metros?\n"
            "Spanish lex        : La longitud de la pista del aeropuerto de Ámsterdam Schiphol es de 3800.0 metros.\n"
            "English answer     : 3800.0\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Cuál es la longitud de la pista del aeropuerto de Ámsterdam Schiphol en metros?"}'
    ),
    # 2. No change needed — synonym verb, correct as-is
    (
        (
            "English question   : When was the Alhambra completed?\n"
            "Spanish translation: ¿Cuándo fue terminada la Alhambra?\n"
            "Spanish lex        : La Alhambra fue completada en 1391.\n"
            "English answer     : 1391\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Cuándo fue terminada la Alhambra?"}'
    ),
    # 3. English proper name used in lex — must keep English form
    (
        (
            "English question   : Who wrote The Catcher in the Rye?\n"
            "Spanish translation: ¿Quién escribió El guardián entre el centeno?\n"
            "Spanish lex        : The Catcher in the Rye fue escrito por J.D. Salinger.\n"
            "English answer     : J.D. Salinger\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Quién escribió The Catcher in the Rye?"}'
    ),
    # 4. Person name translated incorrectly
    (
        (
            "English question   : Who is the mayor of Aarhus?\n"
            "Spanish translation: ¿Quién es el alcalde de Aarhus?\n"
            "Spanish lex        : Jacob Bundsgaard es el alcalde de Aarhus.\n"
            "English answer     : Jacob Bundsgaard\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Quién es el alcalde de Aarhus?"}'
    ),
    # 5. City name form differs
    (
        (
            "English question   : Which country is Aarhus Airport in?\n"
            "Spanish translation: ¿En qué país se encuentra el aeropuerto de Aarhus?\n"
            "Spanish lex        : El aeropuerto de Aarhus da servicio a Aarhus, Dinamarca.\n"
            "English answer     : Denmark\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿En qué país se encuentra el aeropuerto de Aarhus?"}'
    ),
    # 6. Translated name when lex uses original English title
    (
        (
            "English question   : What genre is the album Thriller?\n"
            "Spanish translation: ¿Qué género es el álbum Terror?\n"
            "Spanish lex        : Thriller es un álbum de pop y R&B.\n"
            "English answer     : pop\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Qué género es el álbum Thriller?"}'
    ),
]

# ─────────────────────────────────────────────────────────────
# ANSWER EXTRACTION — separate simple LLM prompt
# ─────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """\
You are a Spanish QA answer extractor for español de España.
Given a Spanish lexicalisation and the English answer, extract the answer as it appears
in the Spanish text (verbatim or near-verbatim).
- If the English answer is "Yes", return "Sí".
- If the English answer is "No", return "No".
- Otherwise, find the closest span in the Spanish text that corresponds to the English answer.

Output JSON only. Schema:
{"answer_es": "..."}"""

ANSWER_FEW_SHOT = [
    (
        (
            "Spanish lex  : El arròs negre es un plato originario de España.\n"
            "English answer: Yes\n\nReturn JSON only. /no_think"
        ),
        '{"answer_es": "Sí"}'
    ),
    (
        (
            "Spanish lex  : La longitud de la pista del aeropuerto de Ámsterdam Schiphol es de 3800.0 metros.\n"
            "English answer: 3800.0\n\nReturn JSON only. /no_think"
        ),
        '{"answer_es": "3800.0"}'
    ),
    (
        (
            "Spanish lex  : Jacob Bundsgaard es el alcalde de Aarhus.\n"
            "English answer: Jacob Bundsgaard\n\nReturn JSON only. /no_think"
        ),
        '{"answer_es": "Jacob Bundsgaard"}'
    ),
    (
        (
            "Spanish lex  : El Acacia Hotel Manila es un hotel de 5 estrellas.\n"
            "English answer: 5-star\n\nReturn JSON only. /no_think"
        ),
        '{"answer_es": "5 estrellas"}'
    ),
]


def log(msg: str):
    print(f"[LOG] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

def load_models():
    log(f"Loading translation pipeline: {TRANSLATION_MODEL}")
    translator = pipeline(
        "translation",
        model=TRANSLATION_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32,
    )

    log(f"Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    llm.generation_config.pad_token_id = tokenizer.pad_token_id
    return translator, tokenizer, llm


# ─────────────────────────────────────────────────────────────
# STEP 1 — TRANSLATION
# ─────────────────────────────────────────────────────────────

def translate_questions(questions: List[str], translator) -> List[str]:
    if not questions:
        return []
    results = translator(questions, max_length=256)
    return [r["translation_text"] for r in results]


# ─────────────────────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────────────────────

def build_llm_prompt(system: str, few_shot: list, user_content: str, tokenizer) -> str:
    messages = [{"role": "system", "content": system}]
    for u, a in few_shot:
        messages.append({"role": "user",      "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def llm_batch(prompts: List[str], tokenizer, llm, max_new_tokens: int = 128) -> List[str]:
    if not prompts:
        return []
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(llm.device)
    output_ids = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)


def try_parse_json(text: str) -> Optional[Dict]:
    # strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ─────────────────────────────────────────────────────────────
# STEP 2 — LLM ENTITY CORRECTION (with retry)
# ─────────────────────────────────────────────────────────────

def correct_questions_batch(
    items: List[Dict],   # each has: en_question, translated_q, lex_es, en_answer
    tokenizer,
    llm,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Returns corrected Spanish questions, one per item."""
    n = len(items)
    results      = [None] * n
    pending_idxs = list(range(n))

    for attempt in range(1, max_retries + 1):
        if not pending_idxs:
            break

        prompts = [
            build_llm_prompt(
                CORRECTOR_SYSTEM,
                CORRECTOR_FEW_SHOT,
                (
                    f"English question   : {items[i]['en_question']}\n"
                    f"Spanish translation: {items[i]['translated_q']}\n"
                    f"Spanish lex        : {items[i]['lex_es']}\n"
                    f"English answer     : {items[i]['en_answer']}\n\n"
                    "Return JSON only. /no_think"
                ),
                tokenizer,
            )
            for i in pending_idxs
        ]

        raw_outputs = llm_batch(prompts, tokenizer, llm)

        still_pending = []
        for local_j, global_i in enumerate(pending_idxs):
            parsed = try_parse_json(raw_outputs[local_j])
            q_es   = (parsed or {}).get("question_es", "").strip()

            if q_es.startswith("¿") and q_es.endswith("?"):
                results[global_i] = q_es
            else:
                still_pending.append(global_i)
                if attempt == max_retries:
                    # fallback: use the raw translation
                    results[global_i] = items[global_i]["translated_q"]
                    tqdm.write(
                        f"  ⚠ correction gave up for: {items[global_i]['en_question']!r} "
                        f"→ using raw translation",
                        file=__import__("sys").stderr,
                    )

        pending_idxs = still_pending
        if still_pending and attempt < max_retries:
            tqdm.write(
                f"  ↻ corrector retry {attempt}/{max_retries} — "
                f"{len(still_pending)} questions still invalid",
                file=__import__("sys").stderr,
            )

    return results


# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM ANSWER EXTRACTION (with retry)
# ─────────────────────────────────────────────────────────────

def extract_answers_batch(
    items: List[Dict],   # each has: lex_es, en_answer
    tokenizer,
    llm,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Returns Spanish answers, one per item."""
    n = len(items)
    results      = [None] * n
    pending_idxs = list(range(n))

    for attempt in range(1, max_retries + 1):
        if not pending_idxs:
            break

        prompts = [
            build_llm_prompt(
                ANSWER_SYSTEM,
                ANSWER_FEW_SHOT,
                (
                    f"Spanish lex  : {items[i]['lex_es']}\n"
                    f"English answer: {items[i]['en_answer']}\n\n"
                    "Return JSON only. /no_think"
                ),
                tokenizer,
            )
            for i in pending_idxs
        ]

        raw_outputs = llm_batch(prompts, tokenizer, llm)

        still_pending = []
        for local_j, global_i in enumerate(pending_idxs):
            parsed = try_parse_json(raw_outputs[local_j])
            a_es   = (parsed or {}).get("answer_es", "").strip()

            if a_es:
                results[global_i] = a_es
            else:
                still_pending.append(global_i)
                if attempt == max_retries:
                    # fallback: use English answer as-is
                    results[global_i] = items[global_i]["en_answer"]
                    tqdm.write(
                        f"  ⚠ answer extraction gave up: {items[global_i]['en_answer']!r} "
                        f"→ keeping English",
                        file=__import__("sys").stderr,
                    )

        pending_idxs = still_pending
        if still_pending and attempt < max_retries:
            tqdm.write(
                f"  ↻ answer retry {attempt}/{max_retries} — "
                f"{len(still_pending)} answers still missing",
                file=__import__("sys").stderr,
            )

    return results


# ─────────────────────────────────────────────────────────────
# XML LOADING
# ─────────────────────────────────────────────────────────────

def parse_webnlg_es_lex(xml_path: str, split: str) -> List[Dict]:
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
        lex_es = [
            lex.text.strip()
            for lex in entry.findall("lex")
            if lex.get("lang") == "es" and lex.text
        ]
        if lex_es:
            entries.append({
                "split":    split,
                "category": entry.get("category", ""),
                "eid":      entry.get("eid", ""),
                "lex_es":   lex_es,
            })
    return entries


def load_es_lex_index(root_dir: str) -> Dict[tuple, List[str]]:
    index: Dict[tuple, List[str]] = {}
    for split in ("train", "dev"):
        split_dir = os.path.join(root_dir, split, "1triples")
        if not os.path.isdir(split_dir):
            log(f"  WARNING: {split}/1triples not found")
            continue
        for xf in sorted(glob.glob(os.path.join(split_dir, "*.xml"))):
            for e in parse_webnlg_es_lex(xf, split):
                index[(e["split"], e["category"], e["eid"])] = e["lex_es"]
    test_dir = os.path.join(root_dir, "test")
    if os.path.isdir(test_dir):
        for xf in sorted(glob.glob(os.path.join(test_dir, "**", "*.xml"), recursive=True)):
            for e in parse_webnlg_es_lex(xf, "test"):
                index[(e["split"], e["category"], e["eid"])] = e["lex_es"]
    log(f"Spanish lex index built: {len(index)} entries")
    return index


# ─────────────────────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "split", "category", "eid", "lex_id",
    "lex_text", "lex_text_es",
    "xml_file", "mtriple", "otriple", "statement",
    "question_idx", "question_type",
    "question", "answer",
    "question_es", "answer_es",
    "generation_errors",
    "corrected_answer", "correction_flag",
    "selected_role", "selection_source",
    "question_translated",   # raw MT output, stored for inspection
]


def load_processed_keys(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            processed.add((row["eid"], row["lex_id"], row["question_idx"]))
    log(f"  Resume: {len(processed)} rows already done — skipping.")
    return processed


def open_csv_writer(output_path: str, write_header: bool):
    f = open(output_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    if write_header:
        w.writeheader()
    return f, w


def flush_rows(writer, csv_file, pending_rows: list):
    for row in pending_rows:
        writer.writerow(row)
    csv_file.flush()
    pending_rows.clear()


# ─────────────────────────────────────────────────────────────
# PRINT SAMPLE
# ─────────────────────────────────────────────────────────────

def print_sample(row: Dict, global_idx: int, n_total: int, elapsed: float):
    errors = row.get("generation_errors", "")
    status = "✅" if not errors else "❌"
    speed  = elapsed / max(global_idx, 1)
    print(f"\n{'─'*65}", flush=True)
    print(f"[{global_idx}/{n_total}] {status}  elapsed={elapsed:.0f}s  "
          f"speed={speed:.2f}s/it  ETA≈{speed*(n_total-global_idx):.0f}s", flush=True)
    print(f"  {row['split']} | {row['category']} | eid={row['eid']} "
          f"lex={row['lex_id']} q={row['question_idx']} [{row['question_type']}]", flush=True)
    print(f"  EN stmt  : {row['statement']}", flush=True)
    print(f"  ES lex   : {row['lex_text_es']}", flush=True)
    print(f"  EN Q     : {row['question']}", flush=True)
    print(f"  MT Q     : {row['question_translated']}", flush=True)
    print(f"  ES Q     : {row['question_es']}", flush=True)
    print(f"  EN A     : {row['answer']}", flush=True)
    print(f"  ES A     : {row['answer_es']}", flush=True)
    if errors:
        print(f"  ⚠ errors : {errors}", flush=True)
    print(f"{'─'*65}", flush=True)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def run_translation(
    tasks: List[Dict],
    translator,
    tokenizer,
    llm,
    output_path: str,
    batch_size:  int = BATCH_SIZE,
    print_every: int = PRINT_EVERY,
    save_every:  int = SAVE_EVERY,
):
    n_total   = len(tasks)
    new_file  = not os.path.exists(output_path)
    csv_file, writer = open_csv_writer(output_path, write_header=new_file)

    pending_rows: List[Dict] = []
    n_ok       = 0
    start_time = time.time()
    global_idx = 0

    bar = tqdm(range(0, n_total, batch_size), desc="Batches",
               unit="batch", dynamic_ncols=True)

    try:
        for bs in bar:
            batch = tasks[bs : bs + batch_size]

            # ── STEP 1: translate EN questions → ES (MarianMT) ─
            raw_translations = translate_questions(
                [t["row"]["question"] for t in batch], translator
            )

            # ── STEP 2: LLM corrects entity names in questions ─
            correction_items = [
                {
                    "en_question":  t["row"]["question"],
                    "translated_q": raw_translations[i],
                    "lex_es":       t["lex_es"],
                    "en_answer":    t["row"]["answer"],
                }
                for i, t in enumerate(batch)
            ]
            corrected_questions = correct_questions_batch(
                correction_items, tokenizer, llm
            )

            # ── STEP 3: LLM extracts Spanish answers ──────────
            answer_items = [
                {"lex_es": t["lex_es"], "en_answer": t["row"]["answer"]}
                for t in batch
            ]
            extracted_answers = extract_answers_batch(
                answer_items, tokenizer, llm
            )

            # ── Assemble rows ─────────────────────────────────
            for i, task in enumerate(batch):
                row    = task["row"]
                lex_es = task["lex_es"]
                q_es   = corrected_questions[i]
                a_es   = extracted_answers[i]

                errors = []
                if not q_es or not q_es.startswith("¿"):
                    errors.append("invalid question_es")
                if not a_es:
                    errors.append("empty answer_es")
                if not errors:
                    n_ok += 1

                out = {
                    **row,
                    "lex_text_es":         lex_es,
                    "question_translated": raw_translations[i],
                    "question_es":         q_es,
                    "answer_es":           a_es,
                    "generation_errors":   "; ".join(errors),
                }
                pending_rows.append(out)
                global_idx += 1

                if global_idx % print_every == 0:
                    print_sample(out, global_idx, n_total, time.time() - start_time)

                if global_idx % save_every == 0:
                    flush_rows(writer, csv_file, pending_rows)
                    tqdm.write(
                        f"[SAVE] {global_idx}/{n_total} — {n_ok} ok",
                        file=__import__("sys").stderr,
                    )

            bar.set_postfix(ok=n_ok, done=global_idx,
                            speed=f"{(time.time()-start_time)/max(global_idx,1):.2f}s/it")
    finally:
        if pending_rows:
            flush_rows(writer, csv_file, pending_rows)
        csv_file.close()

    return n_ok, global_idx


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    translator, tokenizer, llm = load_models()

    log("Building Spanish lexicalisation index from XML…")
    es_lex_index = load_es_lex_index(WEBNLG_ROOT)

    log(f"Loading input CSV: {INPUT_CSV}")
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        qa_rows = list(csv.DictReader(f))
    log(f"  Total QA rows: {len(qa_rows)}")

    processed_keys = load_processed_keys(OUTPUT_CSV)

    tasks: List[Dict] = []
    skipped_done  = 0
    skipped_no_es = 0

    for row in qa_rows:
        key = (row["eid"], row["lex_id"], row["question_idx"])
        if key in processed_keys:
            skipped_done += 1
            continue
        es_key  = (row["split"], row["category"], row["eid"])
        es_list = es_lex_index.get(es_key, [])
        if not es_list:
            skipped_no_es += 1
            continue
        lex_id = int(row["lex_id"]) if row["lex_id"].isdigit() else 0
        lex_es = es_list[min(lex_id, len(es_list) - 1)]
        tasks.append({"row": row, "lex_es": lex_es})

    log(f"  Skipped (already done) : {skipped_done}")
    log(f"  Skipped (no ES lex)    : {skipped_no_es}")
    log(f"  Tasks to process       : {len(tasks)}")

    if not tasks:
        log("Nothing left to do.")
        exit(0)

    start = time.time()
    n_ok, n_total = run_translation(
        tasks, translator, tokenizer, llm,
        output_path=OUTPUT_CSV,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
    )
    
    print("\n" + "=" * 80, flush=True)
    print("COMPLETE", flush=True)
    print(f"  Processed : {n_total} rows in {time.time()-start:.1f}s", flush=True)
    print(f"  Success   : {n_ok}/{n_total}", flush=True)
    print(f"  Output    : {OUTPUT_CSV}", flush=True)
    #print("=" * 80, flush=True){n_missing}", flush=True)
    print(f"  Output    : {OUTPUT_CSV}", flush=True)
    print("=" * 80, flush=True)