import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import glob
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
LLM_MODEL = "Qwen/Qwen3-4B"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ca"  # EN→CA MarianMT
WEBNLG_ROOT = "./WebNLG_CA_BT"

INPUT_CSV = "webnlg_qa_selected_es_v4_validated.csv"
OUTPUT_CSV = "webnlg_qa_selected_es_ca_v4_validated.csv"

BATCH_SIZE = 8
PRINT_EVERY = 25
SAVE_EVERY = 50
MAX_RETRIES = 4
VALIDATE_WITH_LLM = True
ACCEPT_VALIDATOR_FIX_ON_FINAL_ATTEMPT = True
ACCEPT_VALIDATOR_FIX_WHEN_DETERMINISTICALLY_SAFE = True

YES_EQUIVALENTS = {"yes", "sí", "si"}
NO_EQUIVALENTS = {"no"}

NEW_COLUMNS = [
    "lex_text_ca",
    "question_ca",
    "answer_ca",
    "question_ca_translated",
    "ca_generation_errors",
    "ca_validation_status",
    "ca_validation_feedback",
    "ca_generation_attempts",
]

# ─────────────────────────────────────────────────────────────
# STEP 2 — LLM ENTITY CORRECTOR
# ─────────────────────────────────────────────────────────────

CORRECTOR_SYSTEM = """\
You are a post-editor for Catalan QA data in català.
You will receive:
- An English question (original meaning — ground truth)
- A Catalan translation of that question (produced by a machine translator)
- A Catalan lexicalisation of the related fact (reference for entity names and style)
- The English answer (to help identify the entity being asked about)

Your task: correct ONLY what is necessary to make the Catalan question coherent with the lexicalisation.

Rules:
1. ENTITY NAMES ONLY: if the translation uses a Catalan name for an entity but the lexicalisation
   uses a different form (e.g. English name, or different spelling), replace it with the lexicalisation form.
   Example: translation says "L'Alhambra va ser acabada" → lexicalisation says "completada" → BOTH are fine, do NOT change.
   Example: translation says "l'aeroport d'Amsterdam" → lexicalisation says "aeroport de Schiphol" → fix to match.
2. DO NOT change correct synonyms, verb tenses, or grammatical structures that preserve the meaning.
3. DO NOT add or remove information from the question.
4. The question must stay a well-formed Catalan interrogative.
5. If the translation is already coherent with the lexicalisation, return it EXACTLY as-is, with no changes at all.

Output JSON only. Schema:
{"question_ca": "..."}"""

CORRECTOR_FEW_SHOT = [
    (
        (
            "English question    : What is the runway length of Amsterdam Airport Schiphol in metres?\n"
            "Catalan translation: Quina és la longitud de la pista de l'aeroport d'Amsterdam en metres?\n"
            "Catalan lex        : La longitud de la pista de l'aeroport d'Amsterdam Schiphol és de 3800.0 metres.\n"
            "English answer     : 3800.0\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Quina és la longitud de la pista de l\'aeroport d\'Amsterdam Schiphol en metres?"}'
    ),
    (
        (
            "English question    : When was the Alhambra completed?\n"
            "Catalan translation: Quan va ser acabada l'Alhambra?\n"
            "Catalan lex        : L'Alhambra va ser completada el 1391.\n"
            "English answer     : 1391\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Quan va ser acabada l\'Alhambra?"}'
    ),
    (
        (
            "English question    : Who wrote The Catcher in the Rye?\n"
            "Catalan translation: Qui va escriure El vigilant en el camp de sègol?\n"
            "Catalan lex        : The Catcher in the Rye va ser escrit per J.D. Salinger.\n"
            "English answer     : J.D. Salinger\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Qui va escriure The Catcher in the Rye?"}'
    ),
    (
        (
            "English question    : Who is the mayor of Aarhus?\n"
            "Catalan translation: Qui és l'alcalde d'Aarhus?\n"
            "Catalan lex        : Jacob Bundsgaard és l'alcalde d'Aarhus.\n"
            "English answer     : Jacob Bundsgaard\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Qui és l\'alcalde d\'Aarhus?"}'
    ),
    (
        (
            "English question    : Which country is Aarhus Airport in?\n"
            "Catalan translation: En quin país es troba l'aeroport d'Aarhus?\n"
            "Catalan lex        : L'aeroport d'Aarhus dona servei a Aarhus, Dinamarca.\n"
            "English answer     : Denmark\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "En quin país es troba l\'aeroport d\'Aarhus?"}'
    ),
    (
        (
            "English question    : What genre is the album Thriller?\n"
            "Catalan translation: Quin gènere és l'àlbum Terror?\n"
            "Catalan lex        : Thriller és un àlbum de pop i R&B.\n"
            "English answer     : pop\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Quin gènere és l\'àlbum Thriller?"}'
    ),
]

# ─────────────────────────────────────────────────────────────
# ANSWER EXTRACTION — separate simple LLM prompt
# ─────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """\
You are a Catalan QA answer extractor for català.
Given a Catalan lexicalisation and the English answer, extract the answer as it appears
in the Catalan text (verbatim or near-verbatim).
- If the English answer is "Yes", return "Sí".
- If the English answer is "No", return "No".
- Otherwise, find the closest span in the Catalan text that corresponds to the English answer.

Output JSON only. Schema:
{"answer_ca": "..."}"""

ANSWER_FEW_SHOT = [
    (
        (
            "Catalan lex  : L'arròs negre és un plat originari d'Espanya.\n"
            "English answer: Yes\n\nReturn JSON only. /no_think"
        ),
        '{"answer_ca": "Sí"}'
    ),
    (
        (
            "Catalan lex  : La longitud de la pista de l'aeroport d'Amsterdam Schiphol és de 3800.0 metres.\n"
            "English answer: 3800.0\n\nReturn JSON only. /no_think"
        ),
        '{"answer_ca": "3800.0"}'
    ),
    (
        (
            "Catalan lex  : Jacob Bundsgaard és l'alcalde d'Aarhus.\n"
            "English answer: Jacob Bundsgaard\n\nReturn JSON only. /no_think"
        ),
        '{"answer_ca": "Jacob Bundsgaard"}'
    ),
    (
        (
            "Catalan lex  : L'Acacia Hotel Manila és un hotel de 5 estrelles.\n"
            "English answer: 5-star\n\nReturn JSON only. /no_think"
        ),
        '{"answer_ca": "5 estrelles"}'
    ),
]

# ─────────────────────────────────────────────────────────────
# STEP 2/3 — JOINT QA TRANSLATION + EXECUTION-TIME VALIDATION
# ─────────────────────────────────────────────────────────────

QA_GENERATOR_SYSTEM = """\
You are a bilingual QA translator for català.
Translate an English QA pair into Catalan using the Catalan lexicalisation as the
source of truth for entity names, titles, values, and units.

You will receive:
- English statement and triples for context
- English question
- English answer
- Raw MarianMT Catalan question (hint only; it may be wrong)
- Catalan lexicalisation of the fact
- Optional validator feedback from a previous failed attempt

Task: produce a Catalan question and a Catalan answer that form the SAME QA pair.

Rules:
1. Translate QUESTION + ANSWER together. The Catalan answer must answer the Catalan question.
2. Preserve the English question meaning exactly: relation, entity, polarity, number, date,
   unit, comparison, and wh-type (who/when/where/which/how many/etc.).
3. Use the Catalan lexicalisation for entity/title spelling. If the lexicalisation keeps an
   English title/name, keep that title/name in the Catalan question. Do not translate titles
   such as albums, books, airports, organisations, or people unless the lexicalisation does.
4. The Catalan answer should be the closest span in the Catalan lexicalisation. It may include
   the unit when that is the natural answer span (e.g. "3.800,0 metres"). For Yes/No,
   return "Sí" or "No".
5. Do not add facts not present in the English QA pair or Catalan lexicalisation.
6. Return one well-formed Catalan interrogative question and one concise answer.
7. For measurement questions, use natural Catalan even when it does not mirror English word-for-word:
   "How many meters above sea level..." can be "A quina altura sobre el nivell del mar...?".

Output JSON only. Schema:
{"question_ca": "...", "answer_ca": "..."}"""

QA_GENERATOR_FEW_SHOT = [
    (
        (
            "English statement : Amsterdam Airport Schiphol has runway length 3800.0 metres.\n"
            "English question  : What is the runway length of Amsterdam Airport Schiphol in metres?\n"
            "English answer    : 3800.0\n"
            "Raw MT question   : Quina és la longitud de la pista de l'aeroport d'Amsterdam en metres?\n"
            "Catalan lex       : La longitud de la pista de l'aeroport d'Amsterdam Schiphol és de 3800.0 metres.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Quina és la longitud de la pista de l\'aeroport d\'Amsterdam Schiphol en metres?", "answer_ca": "3800.0"}'
    ),
    (
        (
            "English statement : Thriller is an album in the pop genre.\n"
            "English question  : What genre is the album Thriller?\n"
            "English answer    : pop\n"
            "Raw MT question   : Quin gènere és l'àlbum Terror?\n"
            "Catalan lex       : Thriller és un àlbum de pop i R&B.\n"
            "Validator feedback: Title was mistranslated as Terror; keep Thriller.\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Quin gènere és l\'àlbum Thriller?", "answer_ca": "pop"}'
    ),
    (
        (
            "English statement : Aarhus Airport is in Denmark.\n"
            "English question  : Which country is Aarhus Airport in?\n"
            "English answer    : Denmark\n"
            "Raw MT question   : Quin país està l'aeroport d'Aarhus?\n"
            "Catalan lex       : L'aeroport d'Aarhus és a Dinamarca.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "En quin país és l\'aeroport d\'Aarhus?", "answer_ca": "Dinamarca"}'
    ),
    (
        (
            "English statement : The Catcher in the Rye was written by J. D. Salinger.\n"
            "English question  : Who wrote The Catcher in the Rye?\n"
            "English answer    : J. D. Salinger\n"
            "Raw MT question   : Qui va escriure El vigilant en el camp de sègol?\n"
            "Catalan lex       : The Catcher in the Rye va ser escrit per J. D. Salinger.\n"
            "Validator feedback: Keep the title as it appears in the Catalan lex.\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "Qui va escriure The Catcher in the Rye?", "answer_ca": "J. D. Salinger"}'
    ),
    (
        (
            "English statement : Arròs negre is a dish from Spain.\n"
            "English question  : Is arròs negre from Spain?\n"
            "English answer    : Yes\n"
            "Raw MT question   : És arròs negre d'Espanya?\n"
            "Catalan lex       : L'arròs negre és un plat originari d'Espanya.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "L\'arròs negre és originari d\'Espanya?", "answer_ca": "Sí"}'
    ),
    (
        (
            "English statement : Amsterdam Airport Schiphol is -3.3528 metres above sea level.\n"
            "English question  : How many meters above sea level is Schiphol airport?\n"
            "English answer    : -3.3528\n"
            "Raw MT question   : Quants metres sobre el nivell del mar està l'aeroport Schiphol?\n"
            "Catalan lex       : L'aeroport Schiphol és a -3,3528 metres sobre el nivell del mar.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_ca": "A quina altura sobre el nivell del mar és l\'aeroport Schiphol?", "answer_ca": "-3,3528 metres"}'
    ),
]

QA_VALIDATOR_SYSTEM = """\
You are a strict but fair validator for English→Catalan QA translation.

Inputs:
- English question and answer are the ground truth.
- Catalan lexicalisation is the reference for facts, entity/title spelling, units, and values.
- Proposed Catalan question and answer are the candidate outputs.

Decide whether the Catalan QA pair is semantically equivalent and answerable.
Be strict about wrong entity, relation, wh-type, polarity, number/date/unit, title translation,
or an answer that does not answer the Catalan question.
Be fair: do NOT reject because of harmless synonyms, word order, natural Catalan paraphrase,
or verb choices that preserve meaning. Accept Catalan numeric formatting and units when value-equivalent
(e.g. 3.800,0 metres = 3800.0 metres; -3,3528 = -3.3528). Accept natural measurement
questions such as "A quina altura...?" for English "How many meters above sea level...?".

If invalid, give a minimal corrected Catalan QA pair.

Output JSON only. Schema:
{
  "is_valid": true/false,
  "question_ok": true/false,
  "answer_ok": true/false,
  "errors": ["..."],
  "question_ca_fixed": "...",
  "answer_ca_fixed": "..."
}"""

QA_VALIDATOR_FEW_SHOT = [
    (
        (
            "English question : When was the Alhambra completed?\n"
            "English answer   : 1391\n"
            "Catalan lex      : L'Alhambra va ser completada el 1391.\n"
            "Catalan question : Quan va ser acabada l'Alhambra?\n"
            "Catalan answer   : 1391\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_ca_fixed": "Quan va ser acabada l\'Alhambra?", "answer_ca_fixed": "1391"}'
    ),
    (
        (
            "English question : What genre is the album Thriller?\n"
            "English answer   : pop\n"
            "Catalan lex      : Thriller és un àlbum de pop i R&B.\n"
            "Catalan question : Quin gènere és l'àlbum Terror?\n"
            "Catalan answer   : pop\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": false, "question_ok": false, "answer_ok": true, "errors": ["The title Thriller was mistranslated as Terror."], "question_ca_fixed": "Quin gènere és l\'àlbum Thriller?", "answer_ca_fixed": "pop"}'
    ),
    (
        (
            "English question : Who wrote The Catcher in the Rye?\n"
            "English answer   : J. D. Salinger\n"
            "Catalan lex      : The Catcher in the Rye va ser escrit per J. D. Salinger.\n"
            "Catalan question : Què va escriure J. D. Salinger?\n"
            "Catalan answer   : The Catcher in the Rye\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": false, "question_ok": false, "answer_ok": false, "errors": ["The question asks for the work, but the English question asks for the author."], "question_ca_fixed": "Qui va escriure The Catcher in the Rye?", "answer_ca_fixed": "J. D. Salinger"}'
    ),
    (
        (
            "English question : How long is the runway at Amsterdam Airport Schiphol?\n"
            "English answer   : 3800.0\n"
            "Catalan lex      : La longitud de la pista de l'Aeroport Schiphol d'Amsterdam és de 3.800,0 metres.\n"
            "Catalan question : Quina és la longitud de la pista de l'Aeroport Schiphol d'Amsterdam?\n"
            "Catalan answer   : 3.800,0 metres\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_ca_fixed": "Quina és la longitud de la pista de l\'Aeroport Schiphol d\'Amsterdam?", "answer_ca_fixed": "3.800,0 metres"}'
    ),
    (
        (
            "English question : How many meters above sea level is Schiphol airport?\n"
            "English answer   : -3.3528\n"
            "Catalan lex      : L'aeroport Schiphol és a -3,3528 metres sobre el nivell del mar.\n"
            "Catalan question : A quina altura sobre el nivell del mar és l'aeroport Schiphol?\n"
            "Catalan answer   : -3,3528 metres\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_ca_fixed": "A quina altura sobre el nivell del mar és l\'aeroport Schiphol?", "answer_ca_fixed": "-3,3528 metres"}'
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
    return [r["translation_text"].strip() for r in results]


# ─────────────────────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────────────────────

def build_llm_prompt(system: str, few_shot: list, user_content: str, tokenizer) -> str:
    messages = [{"role": "system", "content": system}]
    for u, a in few_shot:
        messages.append({"role": "user", "content": u})
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
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
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
    items: List[Dict],
    tokenizer,
    llm,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Returns corrected Catalan questions, one per item."""
    n = len(items)
    results = [None] * n
    pending_idxs = list(range(n))

    for attempt in range(1, max_retries + 1):
        if not pending_idxs:
            break

        prompts = [
            build_llm_prompt(
                CORRECTOR_SYSTEM,
                CORRECTOR_FEW_SHOT,
                (
                    f"English question    : {items[i]['en_question']}\n"
                    f"Catalan translation: {items[i]['translated_q']}\n"
                    f"Catalan lex        : {items[i]['lex_ca']}\n"
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
            q_ca = (parsed or {}).get("question_ca", "").strip()

            if q_ca.endswith("?") and not q_ca.startswith("¿"):
                results[global_i] = q_ca
            else:
                still_pending.append(global_i)
                if attempt == max_retries:
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

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM ANSWER EXTRACTION (with retry)
# ─────────────────────────────────────────────────────────────

def extract_answers_batch(
    items: List[Dict],
    tokenizer,
    llm,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Returns Catalan answers, one per item."""
    n = len(items)
    results = [None] * n
    pending_idxs = list(range(n))

    for attempt in range(1, max_retries + 1):
        if not pending_idxs:
            break

        prompts = [
            build_llm_prompt(
                ANSWER_SYSTEM,
                ANSWER_FEW_SHOT,
                (
                    f"Catalan lex  : {items[i]['lex_ca']}\n"
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
            a_ca = (parsed or {}).get("answer_ca", "").strip()

            if a_ca:
                results[global_i] = a_ca
            else:
                still_pending.append(global_i)
                if attempt == max_retries:
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

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────
# JOINT QA GENERATION + VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    import unicodedata
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(ch)
    )


def norm_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^\w\s.,'/-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _parse_float_or_none(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def number_variants(token: str) -> List[float]:
    token = (token or "").replace("−", "-").strip().strip(".,")
    if not token or not re.search(r"\d", token):
        return []

    vals = set()
    sign = -1.0 if token.startswith("-") else 1.0
    unsigned = token[1:] if token.startswith("-") else token

    def add(candidate: str):
        v = _parse_float_or_none(candidate)
        if v is not None:
            vals.add(sign * abs(v))

    if "." in unsigned and "," in unsigned:
        last_dot = unsigned.rfind(".")
        last_comma = unsigned.rfind(",")
        if last_comma > last_dot:
            add(unsigned.replace(".", "").replace(",", "."))
        else:
            add(unsigned.replace(",", ""))
    elif unsigned.count(".") > 1:
        add(unsigned.replace(".", ""))
    elif unsigned.count(",") > 1:
        add(unsigned.replace(",", ""))
    elif "." in unsigned or "," in unsigned:
        sep = "." if "." in unsigned else ","
        left, right = unsigned.split(sep, 1)
        add(left + "." + right)
        if len(right) == 3 and len(left) <= 3:
            add(left + right)
    else:
        add(unsigned)

    return sorted(vals)


def extract_number_groups(text: str) -> List[Dict[str, Any]]:
    groups = []
    for m in re.finditer(r"[-−]?\d[\d.,]*", text or ""):
        token = m.group(0).strip().strip(".,")
        variants = number_variants(token)
        if variants:
            groups.append({"token": token, "variants": variants})
    return groups


def extract_numbers(text: str) -> List[str]:
    nums = []
    for g in extract_number_groups(text):
        nums.append(str(g["variants"][0]))
    return nums


def numeric_groups_match(
    expected: Dict[str, Any],
    observed_groups: List[Dict[str, Any]],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-6,
) -> bool:
    for e in expected.get("variants", []):
        for obs in observed_groups:
            for o in obs.get("variants", []):
                if abs(e - o) <= max(abs_tol, rel_tol * max(abs(e), abs(o), 1.0)):
                    return True
    return False


def is_yes_answer(answer: str) -> bool:
    return norm_text(answer) in YES_EQUIVALENTS


def is_no_answer(answer: str) -> bool:
    return norm_text(answer) in NO_EQUIVALENTS


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return norm_text(value) in {"true", "yes", "si", "valid", "ok"}
    return bool(value)


def compact_feedback(errors: List[str], limit: int = 4) -> str:
    if not errors:
        return "none"
    clean = [str(e).strip() for e in errors if str(e).strip()]
    return " | ".join(clean[:limit])


def deterministic_validate_item(item: Dict, q_ca: str, a_ca: str) -> List[str]:
    """Fast guardrails for errors observed in validation.

    These checks are intentionally conservative: semantic equivalence is delegated to
    the LLM validator, while this catches malformed questions, answer emptiness,
    obvious wh-type drift, yes/no drift, and numeric loss.
    """
    errors: List[str] = []
    q_ca = (q_ca or "").strip()
    a_ca = (a_ca or "").strip()
    en_question = item.get("en_question", "")
    en_answer = item.get("en_answer", "")
    lex_ca = item.get("lex_ca", "")

    if not q_ca:
        errors.append("empty question_ca")
    elif q_ca.startswith("¿"):
        errors.append("Catalan question should not use Spanish inverted question punctuation")
    elif not q_ca.endswith("?"):
        errors.append("question_ca must be a Catalan interrogative ending in '?' punctuation")
    if len(q_ca.split()) < 3:
        errors.append("question_ca is too short to be reliable")
    if not a_ca:
        errors.append("empty answer_ca")

    q_norm = norm_text(q_ca)
    enq_norm = norm_text(en_question)
    a_norm = norm_text(a_ca)
    lex_norm = norm_text(lex_ca)

    if re.match(r"^\??\s*(what|which|who|when|where|how)\b", q_norm):
        errors.append("question stem is still in English")

    if enq_norm.startswith("who ") and not re.search(r"\b(qui|persona|nom)\b", q_norm):
        errors.append("question type mismatch: English 'Who' should map to a qui/person-name question")
    if enq_norm.startswith("when ") and not re.search(r"\b(quan|data|any|decada)\b", q_norm):
        errors.append("question type mismatch: English 'When' should ask quan/data/any")
    if enq_norm.startswith("where ") and not re.search(r"\b(on|en quin|en quina|lloc|ubicacio|pais|ciutat|regio|estat)\b", q_norm):
        errors.append("question type mismatch: English 'Where' should ask on/en quin lloc/ubicació")
    if enq_norm.startswith("how many "):
        quantity_form = re.search(r"\b(quant|quants|quantes|nombre|numero|quantitat)\b", q_norm)
        measurement_form = False
        if re.search(
            r"\b(meter|metre|meters|metres|kilometer|kilometre|kilometers|kilometres|mile|miles|foot|feet|inch|inches|above sea level|sea level|runway|length|height|altitude|elevation|wide|deep|area|population|capacity)\b",
            enq_norm,
        ):
            measurement_form = re.search(
                r"\b(a quina altura|altura|altitud|elevacio|longitud|distancia|superficie|area|extensio|capacitat|poblacio|metres|metre|quilometres|milles|peus|nivell del mar|mesura|fa)\b",
                q_norm,
            ) is not None
        if not (quantity_form or measurement_form):
            errors.append("question type mismatch: English 'How many' should ask a quantity or equivalent measurement")
    if enq_norm.startswith("how much ") and not re.search(r"\b(quant|quanta|quants|quantes|quantitat|preu|cost|valor)\b", q_norm):
        errors.append("question type mismatch: English 'How much' should ask quant/quantitat/preu")

    if is_yes_answer(en_answer) and a_norm not in {"si", "sí"}:
        errors.append("English Yes answer must be translated as Sí")
    if is_no_answer(en_answer) and a_norm != "no":
        errors.append("English No answer must be translated as No")

    en_num_groups = extract_number_groups(en_answer)
    if en_num_groups:
        a_num_groups = extract_number_groups(a_ca)
        missing = [g["token"] for g in en_num_groups if not numeric_groups_match(g, a_num_groups)]
        if missing:
            errors.append(f"answer_ca is missing numeric value(s): {', '.join(missing)}")

    if a_ca and not (is_yes_answer(en_answer) or is_no_answer(en_answer)):
        if not en_num_groups and a_norm not in lex_norm:
            if norm_text(en_answer) not in lex_norm:
                errors.append("answer_ca does not appear grounded in the Catalan lexicalisation")

    return errors


def build_qa_generator_prompt(item: Dict, feedback: str, tokenizer) -> str:
    user_content = (
        f"English statement : {item.get('statement', '')}\n"
        f"M-triple          : {item.get('mtriple', '')}\n"
        f"O-triple          : {item.get('otriple', '')}\n"
        f"Question type     : {item.get('question_type', '')}\n"
        f"English question  : {item.get('en_question', '')}\n"
        f"English answer    : {item.get('en_answer', '')}\n"
        f"Raw MT question   : {item.get('raw_translation', '')}\n"
        f"Catalan lex       : {item.get('lex_ca', '')}\n"
        f"Validator feedback: {feedback or 'none'}\n\n"
        "Return JSON only. /no_think"
    )
    return build_llm_prompt(QA_GENERATOR_SYSTEM, QA_GENERATOR_FEW_SHOT, user_content, tokenizer)


def build_qa_validator_prompt(item: Dict, q_ca: str, a_ca: str, tokenizer) -> str:
    user_content = (
        f"English question : {item.get('en_question', '')}\n"
        f"English answer   : {item.get('en_answer', '')}\n"
        f"Question type    : {item.get('question_type', '')}\n"
        f"Catalan lex      : {item.get('lex_ca', '')}\n"
        f"Raw MT question  : {item.get('raw_translation', '')}\n"
        f"Catalan question : {q_ca}\n"
        f"Catalan answer   : {a_ca}\n\n"
        "Return JSON only. /no_think"
    )
    return build_llm_prompt(QA_VALIDATOR_SYSTEM, QA_VALIDATOR_FEW_SHOT, user_content, tokenizer)


def validate_qa_candidates_batch(
    items: List[Dict],
    candidates: List[Dict],
    tokenizer,
    llm,
) -> List[Dict]:
    """Validate candidates with deterministic guardrails and an LLM semantic validator."""
    validations: List[Dict] = []
    prompts: List[str] = []
    prompt_indices: List[int] = []

    for i, cand in enumerate(candidates):
        q_ca = cand.get("question_ca", "")
        a_ca = cand.get("answer_ca", "")
        det_errors = deterministic_validate_item(items[i], q_ca, a_ca)
        validations.append({
            "deterministic_errors": det_errors,
            "validator_errors": [],
            "is_valid": False,
            "question_ok": False,
            "answer_ok": False,
            "question_ca_fixed": "",
            "answer_ca_fixed": "",
            "status": "pending_validator",
        })
        if VALIDATE_WITH_LLM and q_ca and a_ca:
            prompts.append(build_qa_validator_prompt(items[i], q_ca, a_ca, tokenizer))
            prompt_indices.append(i)

    raw_outputs = llm_batch(prompts, tokenizer, llm, max_new_tokens=256) if prompts else []

    for local_j, i in enumerate(prompt_indices):
        parsed = try_parse_json(raw_outputs[local_j]) or {}
        errors = parsed.get("errors", [])
        if isinstance(errors, str):
            errors = [errors]
        validations[i].update({
            "validator_errors": [str(e) for e in errors if str(e).strip()],
            "is_valid": as_bool(parsed.get("is_valid", False)),
            "question_ok": as_bool(parsed.get("question_ok", False)),
            "answer_ok": as_bool(parsed.get("answer_ok", False)),
            "question_ca_fixed": (parsed.get("question_ca_fixed") or "").strip(),
            "answer_ca_fixed": (parsed.get("answer_ca_fixed") or "").strip(),
            "status": "validator_parsed" if parsed else "validator_unparseable",
        })

    for val in validations:
        validator_pass = (not VALIDATE_WITH_LLM) or (
            val["is_valid"] and val["question_ok"] and val["answer_ok"]
        )
        all_errors = val["deterministic_errors"] + ([] if validator_pass else val["validator_errors"])
        val["all_errors"] = all_errors
        val["valid"] = (not all_errors) and validator_pass
        if val["valid"]:
            val["status"] = "valid"
        elif not val["validator_errors"] and VALIDATE_WITH_LLM:
            val["status"] = "invalid_or_unparseable_validator"
        else:
            val["status"] = "invalid"
    return validations


def generate_validated_qa_batch(
    items: List[Dict],
    tokenizer,
    llm,
    max_retries: int = MAX_RETRIES,
) -> List[Dict]:
    """Generate question_ca and answer_ca jointly, validate, and regenerate failures."""
    n = len(items)
    results: List[Optional[Dict]] = [None] * n
    pending_idxs = list(range(n))
    feedback_by_idx = {i: "none" for i in range(n)}
    last_candidate_by_idx: Dict[int, Dict] = {}
    last_validation_by_idx: Dict[int, Dict] = {}

    for attempt in range(1, max_retries + 1):
        if not pending_idxs:
            break

        prompts = [
            build_qa_generator_prompt(items[i], feedback_by_idx.get(i, "none"), tokenizer)
            for i in pending_idxs
        ]
        raw_outputs = llm_batch(prompts, tokenizer, llm, max_new_tokens=256)

        candidates: List[Dict] = []
        for local_j, global_i in enumerate(pending_idxs):
            parsed = try_parse_json(raw_outputs[local_j]) or {}
            q_ca = (parsed.get("question_ca") or "").strip()
            a_ca = (parsed.get("answer_ca") or "").strip()
            cand = {
                "question_ca": q_ca,
                "answer_ca": a_ca,
                "raw_generation": raw_outputs[local_j],
                "parse_ok": bool(parsed),
            }
            candidates.append(cand)
            last_candidate_by_idx[global_i] = cand

        validations = validate_qa_candidates_batch(
            [items[i] for i in pending_idxs], candidates, tokenizer, llm
        )

        still_pending: List[int] = []
        for local_j, global_i in enumerate(pending_idxs):
            cand = candidates[local_j]
            val = validations[local_j]
            last_validation_by_idx[global_i] = val

            if val["valid"]:
                results[global_i] = {
                    "question_ca": cand["question_ca"],
                    "answer_ca": cand["answer_ca"],
                    "validation_status": "valid",
                    "validation_feedback": "",
                    "qa_generation_attempts": attempt,
                    "raw_generation": cand["raw_generation"],
                }
                continue

            errors = val.get("all_errors") or ["QA candidate failed validation"]
            feedback = compact_feedback(errors)
            if val.get("question_ca_fixed") or val.get("answer_ca_fixed"):
                feedback += (
                    f" | Suggested fix: question_ca={val.get('question_ca_fixed', '')!r}; "
                    f"answer_ca={val.get('answer_ca_fixed', '')!r}"
                )
            feedback_by_idx[global_i] = feedback

            fixed_q = val.get("question_ca_fixed", "")
            fixed_a = val.get("answer_ca_fixed", "")
            fixed_errors = deterministic_validate_item(items[global_i], fixed_q, fixed_a) if (fixed_q and fixed_a) else ["no validator fix"]
            if ACCEPT_VALIDATOR_FIX_WHEN_DETERMINISTICALLY_SAFE and not fixed_errors:
                results[global_i] = {
                    "question_ca": fixed_q,
                    "answer_ca": fixed_a,
                    "validation_status": "validator_fixed",
                    "validation_feedback": feedback,
                    "qa_generation_attempts": attempt,
                    "raw_generation": cand["raw_generation"],
                }
                continue

            if attempt < max_retries:
                still_pending.append(global_i)
            else:
                if ACCEPT_VALIDATOR_FIX_ON_FINAL_ATTEMPT and not fixed_errors:
                    results[global_i] = {
                        "question_ca": fixed_q,
                        "answer_ca": fixed_a,
                        "validation_status": "validator_fixed_after_regeneration_failed",
                        "validation_feedback": feedback,
                        "qa_generation_attempts": attempt,
                        "raw_generation": cand["raw_generation"],
                    }
                else:
                    fallback_q = cand.get("question_ca") or items[global_i].get("raw_translation", "")
                    fallback_a = cand.get("answer_ca") or items[global_i].get("en_answer", "")
                    results[global_i] = {
                        "question_ca": fallback_q,
                        "answer_ca": fallback_a,
                        "validation_status": "invalid_after_retries",
                        "validation_feedback": feedback,
                        "qa_generation_attempts": attempt,
                        "raw_generation": cand["raw_generation"],
                    }
                    tqdm.write(
                        f"  ⚠ QA generation gave up after validation: {items[global_i].get('en_question', '')!r} "
                        f"→ {feedback}",
                        file=__import__("sys").stderr,
                    )

        pending_idxs = still_pending
        if still_pending and attempt < max_retries:
            tqdm.write(
                f"  ↻ QA regenerate retry {attempt}/{max_retries} — "
                f"{len(still_pending)} QA pairs still invalid",
                file=__import__("sys").stderr,
            )

    for i, result in enumerate(results):
        if result is None:
            cand = last_candidate_by_idx.get(i, {})
            val = last_validation_by_idx.get(i, {})
            results[i] = {
                "question_ca": cand.get("question_ca") or items[i].get("raw_translation", ""),
                "answer_ca": cand.get("answer_ca") or items[i].get("en_answer", ""),
                "validation_status": "missing_result_fallback",
                "validation_feedback": compact_feedback(val.get("all_errors", ["missing result"])),
                "qa_generation_attempts": max_retries,
                "raw_generation": cand.get("raw_generation", ""),
            }

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────
# XML LOADING
# ─────────────────────────────────────────────────────────────

def parse_webnlg_ca_lex(xml_path: str, split: str) -> List[Dict]:
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
        lex_ca = [
            lex.text.strip()
            for lex in entry.findall("lex")
            if lex.get("lang") == "ca" and lex.text
        ]
        if lex_ca:
            entries.append({
                "split": split,
                "category": entry.get("category", ""),
                "eid": entry.get("eid", ""),
                "lex_ca": lex_ca,
            })
    return entries


def load_ca_lex_index(root_dir: str) -> Dict[tuple, List[str]]:
    index: Dict[tuple, List[str]] = {}
    for split in ("train", "dev"):
        split_dir = os.path.join(root_dir, split, "1triples")
        if not os.path.isdir(split_dir):
            log(f"  WARNING: {split}/1triples not found")
            continue
        for xf in sorted(glob.glob(os.path.join(split_dir, "*.xml"))):
            for e in parse_webnlg_ca_lex(xf, split):
                index[(e["split"], e["category"], e["eid"])] = e["lex_ca"]
    test_dir = os.path.join(root_dir, "test")
    if os.path.isdir(test_dir):
        for xf in sorted(glob.glob(os.path.join(test_dir, "**", "*.xml"), recursive=True)):
            for e in parse_webnlg_ca_lex(xf, "test"):
                index[(e["split"], e["category"], e["eid"])] = e["lex_ca"]
    log(f"Catalan lex index built: {len(index)} entries")
    return index


# ─────────────────────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────────────────────

def read_csv_rows(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def output_fieldnames(input_fieldnames: List[str], output_path: str) -> List[str]:
    if os.path.exists(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
        if existing:
            return existing
    fields = list(input_fieldnames)
    for col in NEW_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def load_processed_keys(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("question_ca", "").strip() and row.get("answer_ca", "").strip():
                processed.add((row.get("eid", ""), row.get("lex_id", ""), row.get("question_idx", "")))
    log(f"  Resume: {len(processed)} rows already done — skipping.")
    return processed


def open_csv_writer(output_path: str, fieldnames: List[str], write_header: bool):
    f = open(output_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    errors = row.get("ca_generation_errors", "")
    status = "✅" if not errors else "❌"
    speed = elapsed / max(global_idx, 1)
    print(f"\n{'─'*65}", flush=True)
    print(
        f"[{global_idx}/{n_total}] {status}  elapsed={elapsed:.0f}s  "
        f"speed={speed:.2f}s/it  ETA≈{speed*(n_total-global_idx):.0f}s",
        flush=True,
    )
    print(
        f"  {row.get('split', '')} | {row.get('category', '')} | eid={row.get('eid', '')} "
        f"lex={row.get('lex_id', '')} q={row.get('question_idx', '')} [{row.get('question_type', '')}]",
        flush=True,
    )
    print(f"  EN stmt  : {row.get('statement', '')}", flush=True)
    print(f"  CA lex   : {row.get('lex_text_ca', '')}", flush=True)
    print(f"  EN Q     : {row.get('question', '')}", flush=True)
    print(f"  MT CA Q  : {row.get('question_ca_translated', '')}", flush=True)
    print(f"  CA Q     : {row.get('question_ca', '')}", flush=True)
    print(f"  EN A     : {row.get('answer', '')}", flush=True)
    print(f"  CA A     : {row.get('answer_ca', '')}", flush=True)
    if row.get("question_es", ""):
        print(f"  ES Q ref : {row.get('question_es', '')}", flush=True)
    if row.get("answer_es", ""):
        print(f"  ES A ref : {row.get('answer_es', '')}", flush=True)
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
    fieldnames: List[str],
    batch_size: int = BATCH_SIZE,
    print_every: int = PRINT_EVERY,
    save_every: int = SAVE_EVERY,
):
    n_total = len(tasks)
    new_file = not os.path.exists(output_path)
    csv_file, writer = open_csv_writer(output_path, fieldnames, write_header=new_file)

    pending_rows: List[Dict] = []
    n_ok = 0
    start_time = time.time()
    global_idx = 0

    bar = tqdm(range(0, n_total, batch_size), desc="Batches", unit="batch", dynamic_ncols=True)

    try:
        for bs in bar:
            batch = tasks[bs : bs + batch_size]

            raw_translations = translate_questions(
                [t["row"]["question"] for t in batch], translator
            )

            qa_items = [
                {
                    "statement": t["row"].get("statement", ""),
                    "mtriple": t["row"].get("mtriple", ""),
                    "otriple": t["row"].get("otriple", ""),
                    "question_type": t["row"].get("question_type", ""),
                    "en_question": t["row"].get("question", ""),
                    "en_answer": t["row"].get("answer", ""),
                    "raw_translation": raw_translations[i],
                    "lex_ca": t["lex_ca"],
                }
                for i, t in enumerate(batch)
            ]
            qa_results = generate_validated_qa_batch(
                qa_items, tokenizer, llm
            )

            for i, task in enumerate(batch):
                row = dict(task["row"])
                lex_ca = task["lex_ca"]
                qa = qa_results[i]
                q_ca = qa["question_ca"]
                a_ca = qa["answer_ca"]

                errors = []
                if qa.get("validation_status") not in {"valid", "validator_fixed", "validator_fixed_after_regeneration_failed"}:
                    errors.append(qa.get("validation_feedback") or qa.get("validation_status") or "QA validation failed")
                if not q_ca or q_ca.startswith("¿") or not q_ca.endswith("?"):
                    errors.append("invalid question_ca")
                if not a_ca:
                    errors.append("empty answer_ca")
                if not errors:
                    n_ok += 1

                out = {
                    **row,
                    "lex_text_ca": lex_ca,
                    "question_ca_translated": raw_translations[i],
                    "question_ca": q_ca,
                    "answer_ca": a_ca,
                    "ca_generation_errors": "; ".join(errors),
                    "ca_validation_status": qa.get("validation_status", ""),
                    "ca_validation_feedback": qa.get("validation_feedback", ""),
                    "ca_generation_attempts": qa.get("qa_generation_attempts", ""),
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

            bar.set_postfix(
                ok=n_ok,
                done=global_idx,
                speed=f"{(time.time()-start_time)/max(global_idx,1):.2f}s/it",
            )
    finally:
        if pending_rows:
            flush_rows(writer, csv_file, pending_rows)
        csv_file.close()

    return n_ok, global_idx


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    translator, tokenizer, llm = load_models()

    log("Building Catalan lexicalisation index from XML…")
    ca_lex_index = load_ca_lex_index(WEBNLG_ROOT)

    log(f"Loading input CSV: {INPUT_CSV}")
    qa_rows = read_csv_rows(INPUT_CSV)
    if not qa_rows:
        raise ValueError(f"Input CSV is empty: {INPUT_CSV}")
    log(f"  Total QA rows: {len(qa_rows)}")

    input_fieldnames = list(qa_rows[0].keys())
    fieldnames = output_fieldnames(input_fieldnames, OUTPUT_CSV)
    processed_keys = load_processed_keys(OUTPUT_CSV)

    tasks: List[Dict] = []
    skipped_done = 0
    skipped_no_ca = 0
    skipped_missing_en_question = 0

    for row in qa_rows:
        key = (row.get("eid", ""), row.get("lex_id", ""), row.get("question_idx", ""))
        if key in processed_keys:
            skipped_done += 1
            continue
        if not row.get("question", "").strip():
            skipped_missing_en_question += 1
            continue

        existing_lex_ca = row.get("lex_text_ca", "").strip()
        if existing_lex_ca:
            lex_ca = existing_lex_ca
        else:
            ca_key = (row.get("split", ""), row.get("category", ""), row.get("eid", ""))
            ca_list = ca_lex_index.get(ca_key, [])
            if not ca_list:
                skipped_no_ca += 1
                continue
            lex_id_raw = row.get("lex_id", "")
            lex_id = int(lex_id_raw) if str(lex_id_raw).isdigit() else 0
            lex_ca = ca_list[min(lex_id, len(ca_list) - 1)]

        tasks.append({"row": row, "lex_ca": lex_ca})

    log(f"  Skipped (already done)       : {skipped_done}")
    log(f"  Skipped (missing EN question): {skipped_missing_en_question}")
    log(f"  Skipped (no CA lex)          : {skipped_no_ca}")
    log(f"  Tasks to process             : {len(tasks)}")

    if not tasks:
        log("Nothing left to do.")
        raise SystemExit(0)

    start = time.time()
    n_ok, n_total = run_translation(
        tasks,
        translator,
        tokenizer,
        llm,
        output_path=OUTPUT_CSV,
        fieldnames=fieldnames,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_every=SAVE_EVERY,
    )

    print("\n" + "=" * 80, flush=True)
    print("COMPLETE", flush=True)
    print(f"  Processed : {n_total} rows in {time.time()-start:.1f}s", flush=True)
    print(f"  Success   : {n_ok}/{n_total}", flush=True)
    print(f"  Input     : {INPUT_CSV}", flush=True)
    print(f"  Output    : {OUTPUT_CSV}", flush=True)
    print("=" * 80, flush=True)