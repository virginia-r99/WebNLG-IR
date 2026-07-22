#!/usr/bin/env python3
"""
Non-LLM punctuation/grammar corrector for English + Spanish QA questions.

What it does:
- normalizes question punctuation:
  - English: "... ?" -> "...?"; ensures final "?"
  - Spanish: ensures opening "¿" and final "?"
- fixes common Spanish interrogative accents at the start of questions:
  - Que -> Qué, Cual -> Cuál, A que -> A qué, etc.
- optionally runs LanguageTool rule-based grammar/punctuation checks.
- applies LanguageTool edits conservatively, avoiding edits that overlap likely entity names,
  titles, acronyms, or numbers.

Install:
    pip install pandas tqdm language_tool_python openpyxl

Local LanguageTool requires Java. For large datasets, local mode is preferred.

Example:
    python non_llm_question_corrector.py \
      --input webnlg_qa_selected_es_v4_validated.csv \
      --output webnlg_qa_selected_es_v4_corrected.csv \
      --en-col question \
      --es-col question_es

Use --overwrite-cols if you want to replace the original columns directly.
Use --apply-spelling only if you are comfortable with spell-check edits; it is off by default
because proper names and titles are often false positives in WebNLG-like data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

# language_tool_python is imported lazily in make_tool() so the deterministic
# normalizers can still be imported/tested without the optional runtime dependency.


# ---------------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------------

SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_outer_quotes(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] in {'"', "'", "“", "‘"} and text[-1] in {'"', "'", "”", "’"}:
        text = text[1:-1].strip()
    return text


def normalize_spaces(text: str) -> str:
    text = CONTROL_RE.sub("", str(text or ""))
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = SPACE_RE.sub(" ", text).strip()

    # Remove spaces before punctuation and after opening inverted punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([¿¡])\s+", r"\1", text)

    # Add a space after punctuation when two words were accidentally glued.
    text = re.sub(r"([,.;:!?])([^\s\d])", r"\1 \2", text)

    # Undo the previous rule for question endings and inverted punctuation.
    text = re.sub(r"\?\s*$", "?", text)
    text = text.replace("¿ ", "¿").replace("¡ ", "¡")
    return text.strip()


def first_alpha_pos(text: str) -> Optional[int]:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return i
    return None


def capitalize_first_alpha(text: str) -> str:
    i = first_alpha_pos(text)
    if i is None:
        return text
    return text[:i] + text[i].upper() + text[i + 1 :]


def remove_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def alnum_tokens(text: str) -> List[str]:
    return re.findall(r"[\wÁÉÍÓÚÜÑáéíóúüñ]+", text, flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Deterministic question punctuation normalization
# ---------------------------------------------------------------------------


def normalize_english_question(text: str) -> str:
    """Normalize English question punctuation only; do not change meaning."""
    q = normalize_spaces(strip_outer_quotes(text))
    if not q:
        return q

    # English questions should not have inverted Spanish punctuation.
    q = re.sub(r"^[¿¡]+\s*", "", q)
    q = re.sub(r"\s*[?？]+\s*$", "", q)
    q = q.strip()
    if q:
        q = capitalize_first_alpha(q)
        q = q + "?"
    return normalize_spaces(q)


ES_INITIAL_QWORDS = {
    "que": "qué",
    "qué": "qué",
    "cual": "cuál",
    "cuál": "cuál",
    "cuales": "cuáles",
    "cuáles": "cuáles",
    "como": "cómo",
    "cómo": "cómo",
    "donde": "dónde",
    "dónde": "dónde",
    "adonde": "adónde",
    "adónde": "adónde",
    "cuando": "cuándo",
    "cuándo": "cuándo",
    "quien": "quién",
    "quién": "quién",
    "quienes": "quiénes",
    "quiénes": "quiénes",
    "cuanto": "cuánto",
    "cuánto": "cuánto",
    "cuanta": "cuánta",
    "cuánta": "cuánta",
    "cuantos": "cuántos",
    "cuántos": "cuántos",
    "cuantas": "cuántas",
    "cuántas": "cuántas",
}

# Leading prepositions commonly used before Spanish interrogatives.
ES_LEADING_PREPS = {
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "sobre", "tras",
}


def _match_case(original: str, replacement_lower: str) -> str:
    if original and original[0].isupper():
        return replacement_lower[0].upper() + replacement_lower[1:]
    return replacement_lower


def fix_initial_spanish_interrogative_accent(text_without_marks: str) -> str:
    """
    Fix only the first interrogative word, optionally after a leading preposition.

    Examples:
      Que país...       -> Qué país...
      En que país...    -> En qué país...
      A que altura...   -> A qué altura...
      Cuantos metros... -> Cuántos metros...
    """
    text = text_without_marks.strip()
    if not text:
        return text

    # Case 1: question starts directly with the interrogative word.
    m = re.match(r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)\b", text)
    if m:
        w = m.group(1)
        key = remove_accents(w).lower()
        if key in ES_INITIAL_QWORDS:
            repl = _match_case(w, ES_INITIAL_QWORDS[key])
            return repl + text[m.end(1) :]

    # Case 2: question starts with a preposition + interrogative word.
    m = re.match(
        r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)(\s+)([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)\b",
        text,
    )
    if m:
        prep, sep, w = m.group(1), m.group(2), m.group(3)
        prep_key = remove_accents(prep).lower()
        q_key = remove_accents(w).lower()
        if prep_key in ES_LEADING_PREPS and q_key in ES_INITIAL_QWORDS:
            repl = _match_case(w, ES_INITIAL_QWORDS[q_key])
            return prep + sep + repl + text[m.end(3) :]

    return text


def normalize_spanish_question(text: str) -> str:
    """Normalize Spanish question punctuation and opening interrogative accent."""
    q = normalize_spaces(strip_outer_quotes(text))
    if not q:
        return q

    # Remove any existing boundary marks, then restore one Spanish pair.
    q = re.sub(r"^[¿?？]+\s*", "", q)
    q = re.sub(r"\s*[?？]+\s*$", "", q)
    q = normalize_spaces(q)

    q = fix_initial_spanish_interrogative_accent(q)
    q = capitalize_first_alpha(q)
    q = f"¿{q}?"

    # Clean up duplicated marks that may have appeared through bad input.
    q = re.sub(r"^¿+", "¿", q)
    q = re.sub(r"\?+$", "?", q)
    q = q.replace("¿¿", "¿")
    return normalize_spaces(q)


# ---------------------------------------------------------------------------
# Conservative LanguageTool application
# ---------------------------------------------------------------------------

# Function words are allowed to change in grammar corrections because they usually
# do not carry dataset entities. This keeps edits such as "a el" -> "al" and
# "que" -> "qué" while avoiding title/entity rewrites.
FUNCTION_WORDS_EN = {
    "a", "an", "the", "of", "in", "on", "at", "to", "from", "for", "with",
    "by", "about", "above", "below", "under", "over", "into", "between", "and",
    "or", "but", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "has", "have", "had", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "many", "much", "long", "far", "old", "high", "tall", "deep",
    "this", "that", "these", "those", "it", "its", "there",
}

FUNCTION_WORDS_ES = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "segun", "según",
    "sin", "sobre", "tras", "el", "la", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "pero", "es", "son", "era", "eran", "fue", "fueron", "ser", "estar",
    "esta", "está", "estan", "están", "se", "su", "sus", "que", "qué", "cual", "cuál",
    "cuales", "cuáles", "quien", "quién", "quienes", "quiénes", "donde", "dónde",
    "cuando", "cuándo", "como", "cómo", "cuanto", "cuánto", "cuanta", "cuánta",
    "cuantos", "cuántos", "cuantas", "cuántas", "si", "sí", "no", "hay",
}

AUTO_SAFE_CATEGORIES = {
    "PUNCTUATION",
    "TYPOGRAPHY",
    "CASING",
    "WHITESPACE",
}

CONDITIONAL_CATEGORIES = {
    "GRAMMAR",
    "MISC",
    "CONFUSED_WORDS",
    "TYPOS",
}

SKIP_RULE_PREFIXES = (
    # Spelling rules are noisy on entity-heavy WebNLG rows; keep off unless --apply-spelling.
    "MORFOLOGIK_RULE",
)


Span = Tuple[int, int]


def spans_overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def protected_spans(text: str) -> List[Span]:
    """Return spans that probably denote entities, titles, acronyms, or numbers."""
    spans: List[Span] = []

    patterns = [
        r"(?<!\w)-?\d+(?:[.,]\d+)*(?!\w)",
        r"\b[A-ZÁÉÍÓÚÜÑ]{2,}(?:[-/][A-ZÁÉÍÓÚÜÑ0-9]+)*\b",
        r"\b[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+[A-Z][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]*\b",
        r"\b(?:[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ'’-]+)"
        r"(?:\s+(?:de|del|la|las|los|el|of|the|and|in|at|for|on|"
        r"[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ'’-]+)){1,}\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))

    single_cap_pat = r"\b[A-ZÁÉÍÓÚÜÑ][a-zÁÉÍÓÚÜÑáéíóúüñ'’-]{2,}\b"

    for m in re.finditer(single_cap_pat, text):
        start = m.start()
        prefix = text[:start].rstrip()

        if not prefix or prefix in {"¿", "¡"}:
            continue

        if prefix[-1] in ".!?¿¡":
            continue

        spans.append((m.start(), m.end()))

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]

    for s, e in spans[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged


def match_attr(match, name: str, default=None):
    return getattr(match, name, default)


def match_category(match) -> str:
    cat = match_attr(match, "category", "")
    if hasattr(cat, "id"):
        cat = cat.id
    return str(cat or "").upper()


def match_rule_id(match) -> str:
    return str(match_attr(match, "ruleId", match_attr(match, "rule_id", "")) or "")


def best_replacement(match) -> Optional[str]:
    reps = match_attr(match, "replacements", []) or []
    if not reps:
        return None
    repl = str(reps[0])
    if "\n" in repl or "\r" in repl or CONTROL_RE.search(repl):
        return None
    return repl


def same_numbers(before: str, after: str) -> bool:
    """Do not allow LanguageTool to change numeric content."""
    nums_before = re.findall(r"-?\d+(?:[.,]\d+)*", before)
    nums_after = re.findall(r"-?\d+(?:[.,]\d+)*", after)
    return nums_before == nums_after


def only_accents_case_or_punct_changed(before: str, after: str) -> bool:
    b = remove_accents(before).lower()
    a = remove_accents(after).lower()
    b = re.sub(r"\W+", "", b, flags=re.UNICODE)
    a = re.sub(r"\W+", "", a, flags=re.UNICODE)
    return b == a


def tokens_are_function_words(tokens: Sequence[str], lang: str) -> bool:
    if not tokens:
        return True
    allowed = FUNCTION_WORDS_ES if lang.startswith("es") else FUNCTION_WORDS_EN
    return all(remove_accents(t).lower() in allowed for t in tokens)


def is_safe_lt_edit(
    text: str,
    offset: int,
    length: int,
    replacement: str,
    match,
    lang: str,
    spans_to_protect: Sequence[Span],
    apply_spelling: bool,
) -> bool:
    before = text[offset : offset + length]
    after = replacement
    rule_id = match_rule_id(match)
    category = match_category(match)

    if not after and category not in AUTO_SAFE_CATEGORIES:
        return False

    if len(after) > max(60, len(before) * 4 + 10):
        return False

    if any(rule_id.startswith(prefix) for prefix in SKIP_RULE_PREFIXES) and not apply_spelling:
        return False
    if category == "TYPOS" and not apply_spelling:
        return False

    edit_span = (offset, offset + length)
    overlaps_protected = any(spans_overlap(edit_span, p) for p in spans_to_protect)

    # Never let LT change numbers.
    if not same_numbers(before, after):
        return False

    # Punctuation/casing/typography edits are safe if they do not alter alphanumeric content.
    if category in AUTO_SAFE_CATEGORIES:
        return only_accents_case_or_punct_changed(before, after) or not re.search(r"\w", before + after)

    if category not in CONDITIONAL_CATEGORIES:
        return False

    # Avoid changing likely entity/title spans.
    if overlaps_protected:
        return False

    if only_accents_case_or_punct_changed(before, after):
        return True

    before_tokens = alnum_tokens(before)
    after_tokens = alnum_tokens(after)

    # Allow small grammar/function-word edits only.
    if tokens_are_function_words(before_tokens + after_tokens, lang):
        return True

    # Very conservative fallback for spelling: one lowercase token -> one lowercase token.
    if apply_spelling and category == "TYPOS":
        if len(before_tokens) == 1 and len(after_tokens) == 1:
            b, a = before_tokens[0], after_tokens[0]
            if b.islower() and a.islower() and len(b) >= 4 and len(a) >= 4:
                return True

    return False


@dataclass
class CorrectionLogItem:
    rule_id: str
    category: str
    before: str
    after: str
    message: str


def apply_languagetool_conservatively(
    text: str,
    tool,
    lang: str,
    apply_spelling: bool = False,
) -> Tuple[str, List[CorrectionLogItem]]:
    """Apply safe LanguageTool corrections while preserving entities/numbers."""
    if not text:
        return text, []

    try:
        matches = tool.check(text)
    except Exception as exc:
        # Keep the row processable even if LT fails on one sentence.
        return text, [CorrectionLogItem("LANGUAGETOOL_ERROR", "ERROR", text, text, str(exc))]

    spans_to_protect = protected_spans(text)
    selected = []
    used_spans: List[Span] = []

    for m in matches:
        offset = int(match_attr(m, "offset", 0) or 0)
        length = int(match_attr(m, "errorLength", match_attr(m, "error_length", 0)) or 0)
        replacement = best_replacement(m)
        if replacement is None:
            continue
        if offset < 0 or length < 0 or offset + length > len(text):
            continue
        edit_span = (offset, offset + length)
        if any(spans_overlap(edit_span, u) for u in used_spans):
            continue
        if not is_safe_lt_edit(
            text=text,
            offset=offset,
            length=length,
            replacement=replacement,
            match=m,
            lang=lang,
            spans_to_protect=spans_to_protect,
            apply_spelling=apply_spelling,
        ):
            continue
        selected.append((offset, length, replacement, m))
        used_spans.append(edit_span)

    # Apply from right to left so offsets remain valid.
    corrected = text
    logs: List[CorrectionLogItem] = []
    for offset, length, replacement, m in sorted(selected, key=lambda x: x[0], reverse=True):
        before = corrected[offset : offset + length]
        corrected = corrected[:offset] + replacement + corrected[offset + length :]
        logs.append(
            CorrectionLogItem(
                rule_id=match_rule_id(m),
                category=match_category(m),
                before=before,
                after=replacement,
                message=str(match_attr(m, "message", "") or ""),
            )
        )

    logs.reverse()
    return corrected, logs


@dataclass
class QuestionCorrectionResult:
    question_en_original: str
    question_es_original: str
    question_en_corrected: str
    question_es_corrected: str
    question_en_changed: bool
    question_es_changed: bool
    question_en_correction_log: str
    question_es_correction_log: str


def correct_question_pair(
    question_en: str,
    question_es: str,
    en_tool,
    es_tool,
    apply_spelling: bool = False,
) -> QuestionCorrectionResult:
    en_original = "" if question_en is None else str(question_en)
    es_original = "" if question_es is None else str(question_es)

    # Pre-normalize punctuation before grammar checks.
    en_pre = normalize_english_question(en_original)
    es_pre = normalize_spanish_question(es_original)

    en_lt, en_log = apply_languagetool_conservatively(
        en_pre, en_tool, lang="en", apply_spelling=apply_spelling
    )
    es_lt, es_log = apply_languagetool_conservatively(
        es_pre, es_tool, lang="es", apply_spelling=apply_spelling
    )

    # Post-normalize because LT may introduce spacing changes.
    en_final = normalize_english_question(en_lt)
    es_final = normalize_spanish_question(es_lt)

    # Add deterministic normalization records if the result changed but LT log is empty.
    if en_final != en_original and not en_log:
        en_log.append(CorrectionLogItem("DETERMINISTIC_NORMALIZATION", "PUNCTUATION", en_original, en_final, "Normalized English question punctuation."))
    if es_final != es_original and not es_log:
        es_log.append(CorrectionLogItem("DETERMINISTIC_NORMALIZATION", "PUNCTUATION", es_original, es_final, "Normalized Spanish question punctuation/accent."))

    return QuestionCorrectionResult(
        question_en_original=en_original,
        question_es_original=es_original,
        question_en_corrected=en_final,
        question_es_corrected=es_final,
        question_en_changed=(en_final != en_original),
        question_es_changed=(es_final != es_original),
        question_en_correction_log=json.dumps([asdict(x) for x in en_log], ensure_ascii=False),
        question_es_correction_log=json.dumps([asdict(x) for x in es_log], ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# DataFrame / file processing
# ---------------------------------------------------------------------------


def make_tool(language: str, remote_server: Optional[str] = None):
    try:
        import language_tool_python
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: language_tool_python. Install with:\n"
            "  pip install language_tool_python\n"
        ) from exc

    if remote_server:
        return language_tool_python.LanguageTool(language, remote_server=remote_server)
    return language_tool_python.LanguageTool(language)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported input extension: {path.suffix}. Use CSV/TSV/XLSX.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {path.suffix}. Use CSV/TSV/XLSX.")


def correct_dataframe(
    df: pd.DataFrame,
    en_col: str,
    es_col: str,
    en_lang: str = "en-US",
    es_lang: str = "es",
    remote_server: Optional[str] = None,
    apply_spelling: bool = False,
    overwrite_cols: bool = False,
) -> pd.DataFrame:
    if en_col not in df.columns:
        raise KeyError(f"English column not found: {en_col}")
    if es_col not in df.columns:
        raise KeyError(f"Spanish column not found: {es_col}")

    out = df.copy()

    with make_tool(en_lang, remote_server=remote_server) as en_tool, make_tool(es_lang, remote_server=remote_server) as es_tool:
        results: List[QuestionCorrectionResult] = []
        for _, row in tqdm(out.iterrows(), total=len(out), desc="Correcting questions", unit="row"):
            results.append(
                correct_question_pair(
                    question_en=row.get(en_col, ""),
                    question_es=row.get(es_col, ""),
                    en_tool=en_tool,
                    es_tool=es_tool,
                    apply_spelling=apply_spelling,
                )
            )

    en_corrected = [r.question_en_corrected for r in results]
    es_corrected = [r.question_es_corrected for r in results]

    if overwrite_cols:
        out[f"{en_col}_before_punct_grammar_correction"] = out[en_col]
        out[f"{es_col}_before_punct_grammar_correction"] = out[es_col]
        out[en_col] = en_corrected
        out[es_col] = es_corrected
    else:
        out[f"{en_col}_corrected"] = en_corrected
        out[f"{es_col}_corrected"] = es_corrected

    out[f"{en_col}_changed"] = [r.question_en_changed for r in results]
    out[f"{es_col}_changed"] = [r.question_es_changed for r in results]
    out[f"{en_col}_correction_log"] = [r.question_en_correction_log for r in results]
    out[f"{es_col}_correction_log"] = [r.question_es_correction_log for r in results]

    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-LLM English/Spanish question punctuation+grammar corrector.")
    parser.add_argument("--input", required=True, help="Input CSV/TSV/XLSX file.")
    parser.add_argument("--output", required=True, help="Output CSV/TSV/XLSX file.")
    parser.add_argument("--en-col", default="question", help="Column containing English questions. Default: question")
    parser.add_argument("--es-col", default="question_es", help="Column containing Spanish questions. Default: question_es")
    parser.add_argument("--en-lang", default="en-US", help="LanguageTool language code for English. Default: en-US")
    parser.add_argument("--es-lang", default="es", help="LanguageTool language code for Spanish. Default: es")
    parser.add_argument("--remote-server", default=None, help="Optional LanguageTool server URL, e.g. http://localhost:8010")
    parser.add_argument("--apply-spelling", action="store_true", help="Also apply safe spelling corrections. Off by default to protect entity names/titles.")
    parser.add_argument("--overwrite-cols", action="store_true", help="Overwrite original question columns instead of adding *_corrected columns.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = read_table(input_path)
    corrected = correct_dataframe(
        df,
        en_col=args.en_col,
        es_col=args.es_col,
        en_lang=args.en_lang,
        es_lang=args.es_lang,
        remote_server=args.remote_server,
        apply_spelling=args.apply_spelling,
        overwrite_cols=args.overwrite_cols,
    )
    write_table(corrected, output_path)

    en_changed_col = f"{args.en_col}_changed"
    es_changed_col = f"{args.es_col}_changed"
    print(f"Wrote: {output_path}")
    print(f"Rows: {len(corrected)}")
    print(f"English changed: {int(corrected[en_changed_col].sum())}")
    print(f"Spanish changed: {int(corrected[es_changed_col].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
