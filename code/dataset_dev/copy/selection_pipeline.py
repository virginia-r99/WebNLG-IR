import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import json
import re
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
WEBNLG_ROOT      = "./WebNLG_ES"
OUTPUT_CSV       = "webnlg_qa_dataset.csv"
CHECKPOINT_EVERY = 50          # generate → judge → write every N entries

GENERATOR_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"

JUDGE_MODELS = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B-Instruct",
    #"microsoft/Phi-4-mini-instruct",
    "google/gemma-3-1b-it",             # instruct variant – has chat template
]

GENERATE_BATCH_SIZE = 4
JUDGE_BATCH_SIZE    = 4

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
GENERATOR_SYSTEM = """\
You are a data generation assistant for question answering.
Your task is to generate natural, answerable questions from a factual statement.
Generate both yes/no questions AND plain extractive QA questions.
Output JSON only. Generate exactly 6 questions: 2 yes/no questions with answer "Yes", \
2 yes/no questions with answer "No", and 2 extractive QA questions (total 6 questions).
Schema: {"statement": "...", "questions": [{"question": "...", "answer": "...", "type": "yes_no" or "extractive"}]}"""

FEW_SHOT_EXAMPLES = [
    {
        "statement": "Arròs negre is from Spain.",
        "output": {
            "statement": "Arròs negre is from Spain.",
            "questions": [
                {"question": "Is Arròs negre from Spain?",            "answer": "Yes",   "type": "yes_no"},
                {"question": "Is Arròs negre a Spanish dish?",        "answer": "Yes",   "type": "yes_no"},
                {"question": "Is Arròs negre from France?",           "answer": "No",    "type": "yes_no"},
                {"question": "Is Arròs negre from Italy?",            "answer": "No",    "type": "yes_no"},
                {"question": "Which country is Arròs negre from?",    "answer": "Spain", "type": "extractive"},
                {"question": "Where does Arròs negre originate from?","answer": "Spain", "type": "extractive"},
            ]
        }
    }
]

JUDGE_SYSTEM = """\
You are a strict QA judge. You receive a statement and one question+answer pair.
Decide:
1. Is the question answerable solely from the statement? (answerable: true/false)
2. Is the provided answer correct given the statement? (correct: true/false)
3. For yes/no questions: if the answer is wrong, what is the correct answer?
   Output the corrected answer as "corrected_answer" (only when type is yes_no
   and correct is false; otherwise omit the field).
Output JSON only – no markdown, no extra text.
Schema:
{"answerable": true, "correct": true}
or (yes/no correction case):
{"answerable": true, "correct": false, "corrected_answer": "Yes"}"""

JUDGE_FEW_SHOT_EXAMPLES = [
    {
        "statement":     "Arròs negre is from Spain.",
        "question_type": "yes_no",
        "question":      "Is Arròs negre from Spain?",
        "answer":        "Yes",
        "output":        {"answerable": True, "correct": True},
    },
    {
        "statement":     "Arròs negre is from Spain.",
        "question_type": "yes_no",
        "question":      "Is Arròs negre from Italy?",
        "answer":        "Yes",
        "output":        {"answerable": True, "correct": False, "corrected_answer": "No"},
    },
    {
        "statement":     "Arròs negre is from Spain.",
        "question_type": "extractive",
        "question":      "Which country is Arròs negre from?",
        "answer":        "Spain",
        "output":        {"answerable": True, "correct": True},
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# WEBNLG LOADING
# ──────────────────────────────────────────────────────────────────────────────
def parse_webnlg_xml(xml_path: str, split: str) -> List[Dict]:
    records = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[WARN] Could not parse {xml_path}: {e}")
        return records
    for entry in tree.findall(".//entry"):
        if entry.get("size") != "1":
            continue
        category = entry.get("category", "")
        eid      = entry.get("eid", "")
        otriples = [t.text.strip() for t in entry.findall(".//originaltripleset/otriple") if t.text]
        mtriples = [t.text.strip() for t in entry.findall(".//modifiedtripleset/mtriple")  if t.text]
        lex_en   = [lex.text.strip() for lex in entry.findall("lex[@lang='en']") if lex.text]
        if not lex_en:
            continue
        for lex_idx, lex_text in enumerate(lex_en):
            records.append({
                "split":     split,
                "category":  category,
                "eid":       eid,
                "lex_id":    lex_idx,
                "xml_file":  Path(xml_path).name,
                "otriple":   " | ".join(otriples),
                "mtriple":   " | ".join(mtriples),
                "statement": lex_text,
            })
    return records


def load_webnlg_size1(root_dir: str) -> List[Dict]:
    all_records = []
    root = Path(root_dir)
    for split in ("train", "dev"):
        split_dir  = root / split
        candidates = list(split_dir.glob("1triples/*.xml")) + list(split_dir.glob("1/*.xml"))
        for xml_path in sorted(candidates):
            all_records.extend(parse_webnlg_xml(str(xml_path), split))
    for xml_path in sorted((root / "test").glob("*.xml")):
        all_records.extend(parse_webnlg_xml(str(xml_path), "test"))
    return all_records

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[LOG] {msg}", flush=True)


def try_parse_json(text: str) -> Optional[Dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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


def is_gemma(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def fmt_content(text: str, model_name: str):
    """Gemma-3 requires content as a list-of-dicts; all others accept plain strings."""
    if is_gemma(model_name):
        return [{"type": "text", "text": text}]
    return text


def build_messages_gen(statement: str, model_name: str) -> List[Dict]:
    """
    One-shot generation messages.
    Gemma-3 has no system role – system prompt is prepended to the first user turn.
    """
    example      = FEW_SHOT_EXAMPLES[0]
    fmt          = lambda t: fmt_content(t, model_name)
    example_json = json.dumps(example["output"], ensure_ascii=False)

    if is_gemma(model_name):
        # Merge system into first user message; no separate system role
        first_user = GENERATOR_SYSTEM + "\n\n" + f"Statement: {example['statement']}\nReturn JSON only."
        return [
            {"role": "user",      "content": fmt(first_user)},
            {"role": "assistant", "content": fmt(example_json)},
            {"role": "user",      "content": fmt(f"Statement: {statement}\nReturn JSON only.")},
        ]
    else:
        return [
            {"role": "system",    "content": fmt(GENERATOR_SYSTEM)},
            {"role": "user",      "content": fmt(f"Statement: {example['statement']}\nReturn JSON only.")},
            {"role": "assistant", "content": fmt(example_json)},
            {"role": "user",      "content": fmt(f"Statement: {statement}\nReturn JSON only.")},
        ]


def build_messages_judge(statement: str, question: str, answer: str,
                         q_type: str, model_name: str) -> List[Dict]:
    """
    One-shot judge messages.
    Gemma-3: no system role – prepend system to first user turn.
    """
    fmt = lambda t: fmt_content(t, model_name)

    def user_text(ex_stmt, ex_qtype, ex_q, ex_a):
        return (f"Statement: {ex_stmt}\nQuestion type: {ex_qtype}\n"
                f"Question: {ex_q}\nAnswer: {ex_a}\nReturn JSON only.")

    real_user = user_text(statement, q_type, question, answer)

    if is_gemma(model_name):
        first_ex   = JUDGE_FEW_SHOT_EXAMPLES[0]
        first_user = JUDGE_SYSTEM + "\n\n" + user_text(
            first_ex["statement"], first_ex["question_type"],
            first_ex["question"],  first_ex["answer"])
        messages = [
            {"role": "user",      "content": fmt(first_user)},
            {"role": "assistant", "content": fmt(json.dumps(first_ex["output"], ensure_ascii=False))},
        ]
        for ex in JUDGE_FEW_SHOT_EXAMPLES[1:]:
            messages.append({"role": "user",      "content": fmt(user_text(
                ex["statement"], ex["question_type"], ex["question"], ex["answer"]))})
            messages.append({"role": "assistant", "content": fmt(json.dumps(ex["output"], ensure_ascii=False))})
        messages.append({"role": "user", "content": fmt(real_user)})
        return messages
    else:
        messages = [{"role": "system", "content": fmt(JUDGE_SYSTEM)}]
        for ex in JUDGE_FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",      "content": fmt(user_text(
                ex["statement"], ex["question_type"], ex["question"], ex["answer"]))})
            messages.append({"role": "assistant", "content": fmt(json.dumps(ex["output"], ensure_ascii=False))})
        messages.append({"role": "user", "content": fmt(real_user)})
        return messages

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / UNLOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_causal_model(model_name: str):
    log(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer._model_name = model_name   # stash for prompt builders

    extra_kwargs = {}
    if "Phi-4" in model_name:
        extra_kwargs["trust_remote_code"]   = True
        extra_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **extra_kwargs,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def unload_model(tokenizer, model):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "split", "category", "eid", "lex_id", "xml_file",
    "otriple", "mtriple", "statement",
    "question_idx", "question_type", "question", "answer", "final_answer",
    "answer_corrected", "judge_answerable", "judge_correct",
    "judge_corrected_answer", "judges_used", "answerable_votes", "correct_votes",
]


def append_rows(rows: List[Dict], path: str):
    if not rows:
        return
    file_exists = Path(path).exists() and Path(path).stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def _apply_template(tokenizer, messages: List[Dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_for_chunk(chunk: List[Dict], tokenizer, model, batch_size: int) -> List[Dict]:
    model_name = getattr(tokenizer, "_model_name", "")
    results = []
    for i in range(0, len(chunk), batch_size):
        batch   = chunk[i : i + batch_size]
        prompts = [_apply_template(tokenizer, build_messages_gen(r["statement"], model_name))
                   for r in batch]
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
        input_len = inputs.input_ids.shape[1]
        decoded   = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        for record, raw_text in zip(batch, decoded):
            parsed    = try_parse_json(raw_text)
            questions = parsed.get("questions", []) if parsed else []
            results.append({**record, "raw_output": raw_text, "questions": questions})
    return results

# ──────────────────────────────────────────────────────────────────────────────
# JUDGING
# ──────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def judge_items(
    items: List[Tuple],
    tokenizer,
    model,
    batch_size: int,
) -> List[Optional[Dict]]:
    model_name = getattr(tokenizer, "_model_name", "")
    verdicts   = []
    for i in range(0, len(items), batch_size):
        batch   = items[i : i + batch_size]
        prompts = [_apply_template(tokenizer,
                                   build_messages_judge(stmt, q, a, qt, model_name))
                   for stmt, q, a, qt in batch]
        inputs  = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_len = inputs.input_ids.shape[1]
        decoded   = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        for raw in decoded:
            verdicts.append(try_parse_json(raw))
    return verdicts


def majority_vote(verdicts: List[Optional[Dict]]) -> Dict:
    valid = [v for v in verdicts if v and isinstance(v, dict)]
    if not valid:
        return {"answerable": False, "correct": False, "corrected_answer": None,
                "judges_used": 0, "answerable_votes": 0, "correct_votes": 0}
    n                = len(valid)
    answerable_votes = sum(1 for v in valid if v.get("answerable", False))
    correct_votes    = sum(1 for v in valid if v.get("correct",    False))
    corrected        = next((v.get("corrected_answer") for v in valid if v.get("corrected_answer")), None)
    return {
        "answerable":       answerable_votes > n / 2,
        "correct":          correct_votes    > n / 2,
        "corrected_answer": corrected,
        "judges_used":      n,
        "answerable_votes": answerable_votes,
        "correct_votes":    correct_votes,
    }


def judge_chunk(
    generated_chunk: List[Dict],
    judge_handles: List[Tuple],   # list of (tokenizer, model)
    batch_size: int,
) -> List[Dict]:
    # Flatten QA pairs from this chunk
    flat_items: List[Tuple] = []
    flat_meta:  List[Dict]  = []
    for rec in generated_chunk:
        for qi, qa in enumerate(rec.get("questions", [])):
            q      = qa.get("question", "")
            a      = qa.get("answer",   "")
            q_type = qa.get("type",     "extractive")
            flat_items.append((rec["statement"], q, a, q_type))
            flat_meta.append({
                "split":    rec["split"],    "category": rec["category"],
                "eid":      rec["eid"],      "lex_id":   rec["lex_id"],
                "xml_file": rec["xml_file"],
                "otriple":  rec["otriple"],  "mtriple":  rec["mtriple"],
                "statement": rec["statement"],
                "question_idx":  qi,
                "question_type": q_type,
                "question": q,
                "answer":   a,
            })

    if not flat_items:
        return []

    per_judge: List[List[Optional[Dict]]] = []
    for tok, mdl in judge_handles:
        verdicts = judge_items(flat_items, tok, mdl, batch_size)
        per_judge.append(verdicts)

    rows = []
    for meta, item_verdicts in zip(flat_meta, zip(*per_judge)):
        agg = majority_vote(list(item_verdicts))

        final_answer = meta["answer"]
        corrected    = False
        if meta["question_type"] == "yes_no" and not agg["correct"] and agg["corrected_answer"]:
            final_answer = agg["corrected_answer"]
            corrected    = True

        rows.append({
            **meta,
            "final_answer":           final_answer,
            "answer_corrected":       corrected,
            "judge_answerable":       agg["answerable"],
            "judge_correct":          agg["correct"],
            "judge_corrected_answer": agg.get("corrected_answer"),
            "judges_used":            agg["judges_used"],
            "answerable_votes":       agg.get("answerable_votes", 0),
            "correct_votes":          agg.get("correct_votes",    0),
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    # 1. Load WebNLG
    log("Loading WebNLG size=1 entries …")
    records = load_webnlg_size1(WEBNLG_ROOT)
    log(f"  → {len(records)} (entry, lex) pairs found")

    # 2. Clean output file
    if Path(OUTPUT_CSV).exists():
        Path(OUTPUT_CSV).unlink()

    # 3. Load generator
    gen_tok, gen_mdl = load_causal_model(GENERATOR_MODEL)

    # 4. Load all judge models once (stay in memory for the whole run)
    judge_handles = []
    for jname in JUDGE_MODELS:
        jtok, jmdl = load_causal_model(jname)
        judge_handles.append((jtok, jmdl))

    # 5. Chunk loop: generate → judge → write to CSV immediately
    all_final  = []
    total_rows = 0
    chunks     = [records[i : i + CHECKPOINT_EVERY]
                  for i in range(0, len(records), CHECKPOINT_EVERY)]

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Chunks")):
        generated_chunk = generate_for_chunk(chunk, gen_tok, gen_mdl, GENERATE_BATCH_SIZE)
        rows            = judge_chunk(generated_chunk, judge_handles, JUDGE_BATCH_SIZE)
        append_rows(rows, OUTPUT_CSV)
        total_rows += len(rows)
        all_final.extend(rows)
        log(f"  ✓ Chunk {chunk_idx + 1}/{len(chunks)} → {len(rows)} rows written "
            f"({total_rows} total so far)")

    # 6. Unload everything
    unload_model(gen_tok, gen_mdl)
    for jtok, jmdl in judge_handles:
        unload_model(jtok, jmdl)

    duration   = time.time() - t0
    total      = len(all_final)
    answerable = sum(1 for r in all_final if r["judge_answerable"])
    correct    = sum(1 for r in all_final if r["judge_correct"])
    corrected  = sum(1 for r in all_final if r["answer_corrected"])

    print("\n" + "=" * 70)
    print(f"DONE in {duration:.1f}s")
    print(f"  Total QA pairs : {total}")
    print(f"  Answerable     : {answerable}  ({100*answerable/max(total,1):.1f}%)")
    print(f"  Correct        : {correct}  ({100*correct/max(total,1):.1f}%)")
    print(f"  Yes/No fixed   : {corrected}")
    print(f"  Output         : {OUTPUT_CSV}")
    print("=" * 70)