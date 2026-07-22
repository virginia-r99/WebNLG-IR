import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import re
import time
import csv
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any

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
OUTPUT_CSV       = "webnlg_qa_selected_es_v4_validated.csv"

BATCH_SIZE   = 8
PRINT_EVERY  = 25
SAVE_EVERY   = 50
MAX_RETRIES  = 4
VALIDATE_WITH_LLM = True
ACCEPT_VALIDATOR_FIX_ON_FINAL_ATTEMPT = True
ACCEPT_VALIDATOR_FIX_WHEN_DETERMINISTICALLY_SAFE = True

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


# ─────────────────────────────────────────────────────────────
# STEP 2/3 — JOINT QA TRANSLATION + EXECUTION-TIME VALIDATION
# ─────────────────────────────────────────────────────────────

QA_GENERATOR_SYSTEM = """\
You are a bilingual QA translator for español de España.
Translate an English QA pair into Spanish using the Spanish lexicalisation as the
source of truth for entity names, titles, values, and units.

You will receive:
- English statement and triples for context
- English question
- English answer
- Raw MarianMT Spanish question (hint only; it may be wrong)
- Spanish lexicalisation of the fact
- Optional validator feedback from a previous failed attempt

Task: produce a Spanish question and a Spanish answer that form the SAME QA pair.

Rules:
1. Translate QUESTION + ANSWER together. The Spanish answer must answer the Spanish question.
2. Preserve the English question meaning exactly: relation, entity, polarity, number, date,
   unit, comparison, and wh-type (who/when/where/which/how many/etc.).
3. Use the Spanish lexicalisation for entity/title spelling. If the lexicalisation keeps an
   English title/name, keep that title/name in the Spanish question. Do not translate titles
   such as albums, books, airports, organisations, or people unless the lexicalisation does.
4. The Spanish answer should be the closest span in the Spanish lexicalisation. It may include
   the unit when that is the natural answer span (e.g. "3.800,0 metros"). For Yes/No,
   return "Sí" or "No".
5. Do not add facts not present in the English QA pair or Spanish lexicalisation.
6. Return one well-formed Spanish interrogative question (¿...?) and one concise answer.
7. For measurement questions, use natural Spanish even when it does not mirror English word-for-word:
   "How many meters above sea level..." can be "¿A qué altura sobre el nivel del mar...?".

Output JSON only. Schema:
{"question_es": "...", "answer_es": "..."}"""

QA_GENERATOR_FEW_SHOT = [
    (
        (
            "English statement : Amsterdam Airport Schiphol has runway length 3800.0 metres.\n"
            "English question  : What is the runway length of Amsterdam Airport Schiphol in metres?\n"
            "English answer    : 3800.0\n"
            "Raw MT question   : ¿Cuál es la longitud de la pista del aeropuerto de Ámsterdam en metros?\n"
            "Spanish lex       : La longitud de la pista del aeropuerto de Ámsterdam Schiphol es de 3800.0 metros.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Cuál es la longitud de la pista del aeropuerto de Ámsterdam Schiphol en metros?", "answer_es": "3800.0"}'
    ),
    (
        (
            "English statement : Thriller is an album in the pop genre.\n"
            "English question  : What genre is the album Thriller?\n"
            "English answer    : pop\n"
            "Raw MT question   : ¿Qué género es el álbum Terror?\n"
            "Spanish lex       : Thriller es un álbum de pop y R&B.\n"
            "Validator feedback: Title was mistranslated as Terror; keep Thriller.\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Qué género es el álbum Thriller?", "answer_es": "pop"}'
    ),
    (
        (
            "English statement : Aarhus Airport is in Denmark.\n"
            "English question  : Which country is Aarhus Airport in?\n"
            "English answer    : Denmark\n"
            "Raw MT question   : ¿Qué país está el aeropuerto de Aarhus?\n"
            "Spanish lex       : El aeropuerto de Aarhus está en Dinamarca.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿En qué país está el aeropuerto de Aarhus?", "answer_es": "Dinamarca"}'
    ),
    (
        (
            "English statement : The Catcher in the Rye was written by J. D. Salinger.\n"
            "English question  : Who wrote The Catcher in the Rye?\n"
            "English answer    : J. D. Salinger\n"
            "Raw MT question   : ¿Quién escribió El guardián entre el centeno?\n"
            "Spanish lex       : The Catcher in the Rye fue escrito por J. D. Salinger.\n"
            "Validator feedback: Keep the title as it appears in the Spanish lex.\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Quién escribió The Catcher in the Rye?", "answer_es": "J. D. Salinger"}'
    ),
    (
        (
            "English statement : Arròs negre is a dish from Spain.\n"
            "English question  : Is arròs negre from Spain?\n"
            "English answer    : Yes\n"
            "Raw MT question   : ¿Es arròs negre de España?\n"
            "Spanish lex       : El arròs negre es un plato originario de España.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿Es el arròs negre originario de España?", "answer_es": "Sí"}'
    ),
    (
        (
            "English statement : Amsterdam Airport Schiphol is -3.3528 metres above sea level.\n"
            "English question  : How many meters above sea level is Schiphol airport?\n"
            "English answer    : -3.3528\n"
            "Raw MT question   : ¿Cuántos metros sobre el nivel del mar está el aeropuerto Schiphol?\n"
            "Spanish lex       : El aeropuerto Schiphol está a -3,3528 metros sobre el nivel del mar.\n"
            "Validator feedback: none\n\n"
            "Return JSON only. /no_think"
        ),
        '{"question_es": "¿A qué altura sobre el nivel del mar está el aeropuerto Schiphol?", "answer_es": "-3,3528 metros"}'
    ),
]

QA_VALIDATOR_SYSTEM = """\
You are a strict but fair validator for English→Spanish QA translation.

Inputs:
- English question and answer are the ground truth.
- Spanish lexicalisation is the reference for facts, entity/title spelling, units, and values.
- Proposed Spanish question and answer are the candidate outputs.

Decide whether the Spanish QA pair is semantically equivalent and answerable.
Be strict about wrong entity, relation, wh-type, polarity, number/date/unit, title translation,
or an answer that does not answer the Spanish question.
Be fair: do NOT reject because of harmless synonyms, word order, natural Spanish paraphrase,
or verb choices that preserve meaning. Accept Spanish numeric formatting and units when value-equivalent
(e.g. 3.800,0 metros = 3800.0 metres; -3,3528 = -3.3528). Accept natural measurement
questions such as "¿A qué altura...?" for English "How many meters above sea level...?".

If invalid, give a minimal corrected Spanish QA pair.

Output JSON only. Schema:
{
  "is_valid": true/false,
  "question_ok": true/false,
  "answer_ok": true/false,
  "errors": ["..."],
  "question_es_fixed": "...",
  "answer_es_fixed": "..."
}"""

QA_VALIDATOR_FEW_SHOT = [
    (
        (
            "English question : When was the Alhambra completed?\n"
            "English answer   : 1391\n"
            "Spanish lex      : La Alhambra fue completada en 1391.\n"
            "Spanish question : ¿Cuándo fue terminada la Alhambra?\n"
            "Spanish answer   : 1391\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_es_fixed": "¿Cuándo fue terminada la Alhambra?", "answer_es_fixed": "1391"}'
    ),
    (
        (
            "English question : What genre is the album Thriller?\n"
            "English answer   : pop\n"
            "Spanish lex      : Thriller es un álbum de pop y R&B.\n"
            "Spanish question : ¿Qué género es el álbum Terror?\n"
            "Spanish answer   : pop\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": false, "question_ok": false, "answer_ok": true, "errors": ["The title Thriller was mistranslated as Terror."], "question_es_fixed": "¿Qué género es el álbum Thriller?", "answer_es_fixed": "pop"}'
    ),
    (
        (
            "English question : Who wrote The Catcher in the Rye?\n"
            "English answer   : J. D. Salinger\n"
            "Spanish lex      : The Catcher in the Rye fue escrito por J. D. Salinger.\n"
            "Spanish question : ¿Qué escribió J. D. Salinger?\n"
            "Spanish answer   : The Catcher in the Rye\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": false, "question_ok": false, "answer_ok": false, "errors": ["The question asks for the work, but the English question asks for the author."], "question_es_fixed": "¿Quién escribió The Catcher in the Rye?", "answer_es_fixed": "J. D. Salinger"}'
    ),
    (
        (
            "English question : How long is the runway at Amsterdam Airport Schiphol?\n"
            "English answer   : 3800.0\n"
            "Spanish lex      : La longitud de la pista del Aeropuerto Schiphol de Ámsterdam es de 3.800,0 metros.\n"
            "Spanish question : ¿Cuál es la longitud de la pista del Aeropuerto Schiphol de Ámsterdam?\n"
            "Spanish answer   : 3.800,0 metros\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_es_fixed": "¿Cuál es la longitud de la pista del Aeropuerto Schiphol de Ámsterdam?", "answer_es_fixed": "3.800,0 metros"}'
    ),
    (
        (
            "English question : How many meters above sea level is Schiphol airport?\n"
            "English answer   : -3.3528\n"
            "Spanish lex      : El aeropuerto Schiphol está a -3,3528 metros sobre el nivel del mar.\n"
            "Spanish question : ¿A qué altura sobre el nivel del mar está el aeropuerto Schiphol?\n"
            "Spanish answer   : -3,3528 metros\n\n"
            "Return JSON only. /no_think"
        ),
        '{"is_valid": true, "question_ok": true, "answer_ok": true, "errors": [], "question_es_fixed": "¿A qué altura sobre el nivel del mar está el aeropuerto Schiphol?", "answer_es_fixed": "-3,3528 metros"}'
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
    text = re.sub(r"[^\w\s.,/-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _parse_float_or_none(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def number_variants(token: str) -> List[float]:
    """Return plausible float interpretations for English/Spanish formatted numbers.

    Handles examples such as:
    - 3800.0 -> 3800.0
    - 3.800,0 -> 3800.0
    - -3,3528 -> -3.3528
    - 1,234 or 1.234 -> both decimal and thousands interpretations when ambiguous
    """
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

    # Both separators: whichever separator appears last is normally the decimal separator.
    if "." in unsigned and "," in unsigned:
        last_dot = unsigned.rfind(".")
        last_comma = unsigned.rfind(",")
        if last_comma > last_dot:
            add(unsigned.replace(".", "").replace(",", "."))  # Spanish style: 3.800,0
        else:
            add(unsigned.replace(",", ""))  # English style: 3,800.0
    elif unsigned.count(".") > 1:
        add(unsigned.replace(".", ""))
    elif unsigned.count(",") > 1:
        add(unsigned.replace(",", ""))
    elif "." in unsigned or "," in unsigned:
        sep = "." if "." in unsigned else ","
        left, right = unsigned.split(sep, 1)
        # Decimal interpretation.
        add(left + "." + right)
        # Thousands interpretation for ambiguous one-separator cases such as 3.800.
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
    """Backward-compatible normalized display of numbers found in text."""
    nums = []
    for g in extract_number_groups(text):
        nums.append(str(g["variants"][0]))
    return nums


def numeric_groups_match(expected: Dict[str, Any], observed_groups: List[Dict[str, Any]], rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> bool:
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


def deterministic_validate_item(item: Dict, q_es: str, a_es: str) -> List[str]:
    """Fast guardrails for errors observed in validation.

    These checks are intentionally conservative: semantic equivalence is delegated to
    the LLM validator, while this catches malformed questions, answer emptiness,
    obvious wh-type drift, yes/no drift, and numeric loss.
    """
    errors: List[str] = []
    q_es = (q_es or "").strip()
    a_es = (a_es or "").strip()
    en_question = item.get("en_question", "")
    en_answer = item.get("en_answer", "")
    lex_es = item.get("lex_es", "")

    if not q_es:
        errors.append("empty question_es")
    elif not (q_es.startswith("¿") and q_es.endswith("?")):
        errors.append("question_es must be a Spanish interrogative wrapped in ¿...? punctuation")
    if len(q_es.split()) < 3:
        errors.append("question_es is too short to be reliable")
    if not a_es:
        errors.append("empty answer_es")

    q_norm = norm_text(q_es)
    enq_norm = norm_text(en_question)
    a_norm = norm_text(a_es)
    lex_norm = norm_text(lex_es)

    # Obvious untranslated question stems. This avoids rejecting English titles inside otherwise Spanish questions.
    if re.match(r"^¿?\s*(what|which|who|when|where|how)\b", q_norm):
        errors.append("question stem is still in English")

    # Conservative wh-type checks. These catch many real relation inversions without
    # rejecting natural Spanish variants.
    if enq_norm.startswith("who ") and not re.search(r"\b(quien|quienes|nombre|persona)\b", q_norm):
        errors.append("question type mismatch: English 'Who' should map to a quién/quiénes/person-name question")
    if enq_norm.startswith("when ") and not re.search(r"\b(cuando|fecha|ano|año|decada|década)\b", q_norm):
        errors.append("question type mismatch: English 'When' should ask cuándo/fecha/año")
    if enq_norm.startswith("where ") and not re.search(r"\b(donde|dónde|en que|en qué|ubicacion|ubicación|lugar|pais|país|ciudad|region|región|estado)\b", q_norm):
        errors.append("question type mismatch: English 'Where' should ask dónde/en qué lugar/ubicación")
    if enq_norm.startswith("how many "):
        quantity_form = re.search(r"\b(cuantos|cuantas|cuanto|cuanta|numero|número|cantidad)\b", q_norm)
        measurement_form = False
        # Natural Spanish often turns measurement questions into "¿A qué altura...?",
        # "¿Cuál es la longitud...?", etc. This is equivalent for questions like
        # "How many meters above sea level...?" or "How long...?".
        if re.search(r"\b(meter|metre|meters|metres|kilometer|kilometre|kilometers|kilometres|mile|miles|foot|feet|inch|inches|above sea level|sea level|runway|length|height|altitude|elevation|wide|deep|area|population|capacity)\b", enq_norm):
            measurement_form = re.search(
                r"\b(a que altura|a qué altura|altura|altitud|elevacion|elevación|longitud|distancia|superficie|area|área|extension|extensión|capacidad|poblacion|población|metros|metro|kilometros|kilómetros|millas|pies|nivel del mar|mide|medida)\b",
                q_norm,
            ) is not None
        if not (quantity_form or measurement_form):
            errors.append("question type mismatch: English 'How many' should ask a quantity or equivalent measurement")
    if enq_norm.startswith("how much ") and not re.search(r"\b(cuanto|cuanta|cuantos|cuantas|cantidad|precio|coste|valor)\b", q_norm):
        errors.append("question type mismatch: English 'How much' should ask cuánto/cantidad/precio")

    # Yes/No answers must be normalised, not copied from English.
    if is_yes_answer(en_answer) and a_norm not in {"si", "sí"}:
        errors.append("English Yes answer must be translated as Sí")
    if is_no_answer(en_answer) and a_norm != "no":
        errors.append("English No answer must be translated as No")

    # Preserve numeric values in answers, accepting English and Spanish numeric formatting.
    en_num_groups = extract_number_groups(en_answer)
    if en_num_groups:
        a_num_groups = extract_number_groups(a_es)
        missing = [g["token"] for g in en_num_groups if not numeric_groups_match(g, a_num_groups)]
        if missing:
            errors.append(f"answer_es is missing numeric value(s): {', '.join(missing)}")

    # For non-boolean answers, the extracted Spanish answer should usually be grounded
    # in the Spanish lexicalisation. Use as an execution guard, but after numeric checks
    # to avoid false positives caused by units/paraphrases.
    if a_es and not (is_yes_answer(en_answer) or is_no_answer(en_answer)):
        if not en_num_groups and a_norm not in lex_norm:
            # Many entity names are identical across languages. If neither the Spanish
            # answer nor English answer appears in the lex, the answer is likely ungrounded.
            if norm_text(en_answer) not in lex_norm:
                errors.append("answer_es does not appear grounded in the Spanish lexicalisation")

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
        f"Spanish lex       : {item.get('lex_es', '')}\n"
        f"Validator feedback: {feedback or 'none'}\n\n"
        "Return JSON only. /no_think"
    )
    return build_llm_prompt(QA_GENERATOR_SYSTEM, QA_GENERATOR_FEW_SHOT, user_content, tokenizer)


def build_qa_validator_prompt(item: Dict, q_es: str, a_es: str, tokenizer) -> str:
    user_content = (
        f"English question : {item.get('en_question', '')}\n"
        f"English answer   : {item.get('en_answer', '')}\n"
        f"Question type    : {item.get('question_type', '')}\n"
        f"Spanish lex      : {item.get('lex_es', '')}\n"
        f"Raw MT question  : {item.get('raw_translation', '')}\n"
        f"Spanish question : {q_es}\n"
        f"Spanish answer   : {a_es}\n\n"
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
        q_es = cand.get("question_es", "")
        a_es = cand.get("answer_es", "")
        det_errors = deterministic_validate_item(items[i], q_es, a_es)
        validations.append({
            "deterministic_errors": det_errors,
            "validator_errors": [],
            "is_valid": False,
            "question_ok": False,
            "answer_ok": False,
            "question_es_fixed": "",
            "answer_es_fixed": "",
            "status": "pending_validator",
        })
        if VALIDATE_WITH_LLM and q_es and a_es:
            prompts.append(build_qa_validator_prompt(items[i], q_es, a_es, tokenizer))
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
            "question_es_fixed": (parsed.get("question_es_fixed") or "").strip(),
            "answer_es_fixed": (parsed.get("answer_es_fixed") or "").strip(),
            "status": "validator_parsed" if parsed else "validator_unparseable",
        })

    for i, val in enumerate(validations):
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
    """Generate question_es and answer_es jointly, validate, and regenerate failures.

    Returns per-item dicts with question_es, answer_es, validation status, feedback,
    attempt count, and raw generation output.
    """
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
            q_es = (parsed.get("question_es") or "").strip()
            a_es = (parsed.get("answer_es") or "").strip()
            cand = {
                "question_es": q_es,
                "answer_es": a_es,
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
                    "question_es": cand["question_es"],
                    "answer_es": cand["answer_es"],
                    "validation_status": "valid",
                    "validation_feedback": "",
                    "qa_generation_attempts": attempt,
                    "raw_generation": cand["raw_generation"],
                }
                continue

            errors = val.get("all_errors") or ["QA candidate failed validation"]
            feedback = compact_feedback(errors)
            # Give the validator's corrected pair back to the generator as concrete guidance.
            if val.get("question_es_fixed") or val.get("answer_es_fixed"):
                feedback += (
                    f" | Suggested fix: question_es={val.get('question_es_fixed', '')!r}; "
                    f"answer_es={val.get('answer_es_fixed', '')!r}"
                )
            feedback_by_idx[global_i] = feedback

            # If the validator already supplied a deterministically safe fix, accept it
            # immediately instead of spending all retry attempts regenerating the same error.
            fixed_q = val.get("question_es_fixed", "")
            fixed_a = val.get("answer_es_fixed", "")
            fixed_errors = deterministic_validate_item(items[global_i], fixed_q, fixed_a) if (fixed_q and fixed_a) else ["no validator fix"]
            if ACCEPT_VALIDATOR_FIX_WHEN_DETERMINISTICALLY_SAFE and not fixed_errors:
                results[global_i] = {
                    "question_es": fixed_q,
                    "answer_es": fixed_a,
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
                        "question_es": fixed_q,
                        "answer_es": fixed_a,
                        "validation_status": "validator_fixed_after_regeneration_failed",
                        "validation_feedback": feedback,
                        "qa_generation_attempts": attempt,
                        "raw_generation": cand["raw_generation"],
                    }
                else:
                    # Keep the last candidate for inspection, but mark the row as invalid.
                    fallback_q = cand.get("question_es") or items[global_i].get("raw_translation", "")
                    fallback_a = cand.get("answer_es") or items[global_i].get("en_answer", "")
                    results[global_i] = {
                        "question_es": fallback_q,
                        "answer_es": fallback_a,
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

    # Safety fallback, should rarely be reached.
    for i, result in enumerate(results):
        if result is None:
            cand = last_candidate_by_idx.get(i, {})
            val = last_validation_by_idx.get(i, {})
            results[i] = {
                "question_es": cand.get("question_es") or items[i].get("raw_translation", ""),
                "answer_es": cand.get("answer_es") or items[i].get("en_answer", ""),
                "validation_status": "missing_result_fallback",
                "validation_feedback": compact_feedback(val.get("all_errors", ["missing result"])),
                "qa_generation_attempts": max_retries,
                "raw_generation": cand.get("raw_generation", ""),
            }

    return results  # type: ignore[return-value]


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
    "validation_status",
    "validation_feedback",
    "qa_generation_attempts",
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

            # ── STEP 2/3: jointly translate QA, validate, and regenerate failures ─
            qa_items = [
                {
                    "statement":       t["row"].get("statement", ""),
                    "mtriple":         t["row"].get("mtriple", ""),
                    "otriple":         t["row"].get("otriple", ""),
                    "question_type":   t["row"].get("question_type", ""),
                    "en_question":     t["row"]["question"],
                    "en_answer":       t["row"]["answer"],
                    "raw_translation": raw_translations[i],
                    "lex_es":          t["lex_es"],
                }
                for i, t in enumerate(batch)
            ]
            qa_results = generate_validated_qa_batch(
                qa_items, tokenizer, llm
            )

            # ── Assemble rows ─────────────────────────────────
            for i, task in enumerate(batch):
                row    = task["row"]
                lex_es = task["lex_es"]
                qa     = qa_results[i]
                q_es   = qa["question_es"]
                a_es   = qa["answer_es"]

                errors = []
                if qa.get("validation_status") not in {"valid", "validator_fixed", "validator_fixed_after_regeneration_failed"}:
                    errors.append(qa.get("validation_feedback") or qa.get("validation_status") or "QA validation failed")
                if not q_es or not q_es.startswith("¿"):
                    errors.append("invalid question_es")
                if not a_es:
                    errors.append("empty answer_es")
                if not errors:
                    n_ok += 1

                out = {
                    **row,
                    "lex_text_es":             lex_es,
                    "question_translated":     raw_translations[i],
                    "question_es":             q_es,
                    "answer_es":               a_es,
                    "generation_errors":       "; ".join(errors),
                    "validation_status":       qa.get("validation_status", ""),
                    "validation_feedback":     qa.get("validation_feedback", ""),
                    "qa_generation_attempts":  qa.get("qa_generation_attempts", ""),
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