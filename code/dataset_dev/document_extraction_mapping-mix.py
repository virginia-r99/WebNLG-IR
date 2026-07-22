import os
import csv
import glob
import random
import hashlib
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

WEBNLG_ROOT = "./WebNLG_ES"

QA_CSV = "webnlg_qa_selected_es_v4_validated.csv"

# Mixed triples + English verbalisation only.
OUT_QA = "webnlg_qa_selected_es_v4_validated_mixed_triples_text_en_min2.csv"
OUT_TRIPLESETS = "ir_triplesets_mixed_triples_text_en_min2.csv"
OUT_TEXTS = "ir_texts_mixed_triples_text_en_min2.csv"
OUT_REL_TRIPLE = "ir_relations_triplesets_mixed_triples_text_en_min2.csv"
OUT_REL_TEXT = "ir_relations_texts_mixed_triples_text_en_min2.csv"

RANDOM_SEED = 42
TRIPLE_SEP = " ||| "

MIN_TRIPLES = 2

# Both mixed triples and verbalised text are labelled as English documents.
DOC_LANG_LABEL = "en"

# Requirements:
#   - entry size >= 2
#   - EN triples exist
#   - ES triples exist
#   - EN and ES triplesets have the same number of triples
#   - EN lexicalisation exists
#
# ES lexicalisation is NOT required because this version verbalises only in English.
REQUIRE_PARALLEL_REPRESENTATIONS = True

random.seed(RANDOM_SEED)


def log(msg):
    print(f"[LOG] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# NORMALIZATION / IDS
# ─────────────────────────────────────────────────────────────

def normalise_triple(t: str) -> str:
    return " ".join(str(t).lower().strip().split())


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def make_entry_key_tuple(entry: Dict) -> Tuple[str, str, str, int]:
    return (
        entry["split"],
        entry["category"],
        entry["eid"],
        int(entry["size"]),
    )


def make_entry_key_str_from_tuple(key: Tuple[str, str, str, int]) -> str:
    split, category, eid, size = key
    return f"{split}__{category}__{eid}__{size}triples"


def make_entry_key_str(entry: Dict) -> str:
    return make_entry_key_str_from_tuple(make_entry_key_tuple(entry))


def make_tripleset_doc_id(base_id: str, lang: str = DOC_LANG_LABEL) -> str:
    """
    One mixed-language tripleset document per base entry.

    The mixed tripleset itself contains EN and ES triples,
    but the document is labelled as lang='en' for compatibility with the
    previous pipeline and because the paired verbalisation is English.
    """
    return f"tripleset__{base_id}__{lang}"


def make_text_doc_id(base_id: str, lang: str = DOC_LANG_LABEL) -> str:
    """
    One English text document per base entry.
    """
    return f"text__{base_id}__{lang}"


def stable_choice(items: List[str], key: str) -> Tuple[int, str]:
    """
    Deterministic pseudo-random choice, stable across runs and machines.
    """
    if not items:
        raise ValueError("stable_choice received an empty list.")

    digest = hashlib.md5(f"{RANDOM_SEED}::{key}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(items)
    return idx, items[idx]


def stable_rng(key: str) -> random.Random:
    """
    Deterministic RNG per entry.
    """
    digest = hashlib.md5(f"{RANDOM_SEED}::{key}".encode("utf-8")).hexdigest()
    seed = int(digest, 16) % (2**32)
    return random.Random(seed)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"  Saved {len(rows):>7} rows → {path}")


def write_filtered_qa_csv(path: str, original_qa_rows: List[Dict], covered_mtriples: set):
    """
    Keep only QA rows whose mtriple is covered by both:
      - the final mixed tripleset collection
      - the final English text collection
    """
    if not original_qa_rows:
        raise ValueError("No QA rows to write.")

    fieldnames = list(original_qa_rows[0].keys())

    kept_rows = [
        row for row in original_qa_rows
        if normalise_triple(row.get("mtriple", "")) in covered_mtriples
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)

    log(f"  Saved {len(kept_rows):>7} filtered QA rows → {path}")

    return kept_rows


# ─────────────────────────────────────────────────────────────
# MIXED TRIPLESET CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def make_mixed_tripleset(entry: Dict) -> Tuple[List[str], List[str]]:
    """
    Build a mixed-language tripleset from aligned EN/ES triplesets.

    Assumption:
      The English and Spanish triplesets are aligned by position:
      mtriples[i] and striples[i] are alternative language realisations
      of the same triple/fact.

    This preserves the original tripleset meaning:
      - no triples are moved across entries
      - no facts are shuffled
      - the original triple order is preserved
      - only the language of each triple position is selected

    For n triples:
      - if n is even, use n/2 EN and n/2 ES.
      - if n is odd, use floor(n/2) from one language and ceil(n/2) from the other.
        The language receiving the extra triple is chosen randomly per entry,
        but deterministically using RANDOM_SEED + base_id.

    Returns:
      mixed_triples: list of selected triples in original tripleset order.
      mixed_lang_pattern: list of "en" or "es" for each triple position.
    """
    base_id = make_entry_key_str(entry)
    mtriples = entry["mtriples"]
    striples = entry["striples"]

    n_en = len(mtriples)
    n_es = len(striples)

    if n_en != n_es:
        raise ValueError(
            f"Cannot mix entry {base_id}: English and Spanish triplesets have "
            f"different numbers of triples: n_en_triples={n_en} != n_es_triples={n_es}"
        )

    n = n_en

    if n < MIN_TRIPLES:
        raise ValueError(f"Cannot mix entry {base_id}: tripleset size {n} < {MIN_TRIPLES}")

    rng = stable_rng(f"mix::{base_id}")

    if n % 2 == 0:
        target_n_en = n // 2
        target_n_es = n // 2
    else:
        if rng.random() < 0.5:
            target_n_en = (n // 2) + 1
            target_n_es = n // 2
        else:
            target_n_en = n // 2
            target_n_es = (n // 2) + 1

    en_positions = set(rng.sample(range(n), target_n_en))

    mixed_triples = []
    mixed_lang_pattern = []

    for i in range(n):
        if i in en_positions:
            mixed_triples.append(mtriples[i])
            mixed_lang_pattern.append("en")
        else:
            mixed_triples.append(striples[i])
            mixed_lang_pattern.append("es")

    assert mixed_lang_pattern.count("en") == target_n_en
    assert mixed_lang_pattern.count("es") == target_n_es
    assert len(mixed_triples) == n

    return mixed_triples, mixed_lang_pattern


# ─────────────────────────────────────────────────────────────
# PARSING WEBNLG XML
# ─────────────────────────────────────────────────────────────

def parse_webnlg_xml_full(xml_path: str, split: str) -> List[Dict]:
    entries = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        log(f"  XML parse error in {xml_path}: {e}")
        return entries

    for entry in root.iter("entry"):
        category = entry.get("category", "")
        eid = entry.get("eid", "")
        size_raw = entry.get("size", "1")

        mtriples = [
            mt.text.strip()
            for mt in entry.findall("./modifiedtripleset/mtriple")
            if mt.text and mt.text.strip()
        ]

        striples = [
            st.text.strip()
            for st in entry.findall("./spanishtripleset/striple")
            if st.text and st.text.strip()
        ]

        if not mtriples:
            mtriples = [
                mt.text.strip()
                for mt in entry.iter("mtriple")
                if mt.text and mt.text.strip()
            ]

        if not striples:
            striples = [
                st.text.strip()
                for st in entry.iter("striple")
                if st.text and st.text.strip()
            ]

        if not mtriples:
            continue

        lex_en = [
            lex.text.strip()
            for lex in entry.findall("lex")
            if lex.get("lang") in ("en", None) and lex.text and lex.text.strip()
        ]

        lex_es = [
            lex.text.strip()
            for lex in entry.findall("lex")
            if lex.get("lang") == "es" and lex.text and lex.text.strip()
        ]

        size = safe_int(size_raw, default=len(mtriples))

        entries.append({
            "split": split,
            "category": category,
            "eid": eid,
            "size": size,
            "xml_file": os.path.basename(xml_path),
            "mtriples": mtriples,
            "striples": striples,
            "lex_en": lex_en,
            "lex_es": lex_es,
        })

    return entries


def load_webnlg_all_sizes(root_dir: str) -> List[Dict]:
    all_entries = []

    for split in ("train", "dev"):
        split_dir = os.path.join(root_dir, split)

        if not os.path.isdir(split_dir):
            log(f"  WARNING: split dir not found: {split_dir}")
            continue

        for n in range(1, 8):
            size_dir = os.path.join(split_dir, f"{n}triples")

            if not os.path.isdir(size_dir):
                continue

            xml_files = sorted(glob.glob(os.path.join(size_dir, "*.xml")))
            log(f"  {split}/{n}triples — {len(xml_files)} XML file(s)")

            for xf in xml_files:
                all_entries.extend(parse_webnlg_xml_full(xf, split))

    test_dir = os.path.join(root_dir, "test")

    if os.path.isdir(test_dir):
        xml_files = sorted(
            glob.glob(os.path.join(test_dir, "**", "*.xml"), recursive=True)
        )
        log(f"  test/ — {len(xml_files)} XML file(s)")

        for xf in xml_files:
            all_entries.extend(parse_webnlg_xml_full(xf, "test"))
    else:
        log("  WARNING: test/ folder not found")

    log(f"Total raw entries loaded: {len(all_entries)}")
    return all_entries


# ─────────────────────────────────────────────────────────────
# MERGE / FILTER ENTRIES
# ─────────────────────────────────────────────────────────────

def merge_entries_by_base_key(entries: List[Dict]) -> List[Dict]:
    merged = {}
    collision_diagnostics = []

    for entry in entries:
        key = make_entry_key_tuple(entry)

        if key not in merged:
            merged[key] = {
                "split": entry["split"],
                "category": entry["category"],
                "eid": entry["eid"],
                "size": entry["size"],
                "xml_file": entry["xml_file"],
                "xml_files_seen": {entry["xml_file"]},
                "mtriples": list(entry["mtriples"]),
                "striples": list(entry["striples"]),
                "lex_en": list(entry["lex_en"]),
                "lex_es": list(entry["lex_es"]),
            }
            continue

        current = merged[key]
        current["xml_files_seen"].add(entry["xml_file"])

        if list(entry["mtriples"]) != list(current["mtriples"]):
            collision_diagnostics.append({
                "base_id": make_entry_key_str_from_tuple(key),
                "field": "mtriples",
                "kept": TRIPLE_SEP.join(current["mtriples"]),
                "seen": TRIPLE_SEP.join(entry["mtriples"]),
                "xml_file": entry["xml_file"],
            })

        if entry["striples"] and not current["striples"]:
            current["striples"] = list(entry["striples"])
        elif entry["striples"] and list(entry["striples"]) != list(current["striples"]):
            collision_diagnostics.append({
                "base_id": make_entry_key_str_from_tuple(key),
                "field": "striples",
                "kept": TRIPLE_SEP.join(current["striples"]),
                "seen": TRIPLE_SEP.join(entry["striples"]),
                "xml_file": entry["xml_file"],
            })

        current["lex_en"].extend(entry["lex_en"])
        current["lex_es"].extend(entry["lex_es"])

    out = []

    for key, entry in merged.items():
        entry["lex_en"] = list(dict.fromkeys(entry["lex_en"]))
        entry["lex_es"] = list(dict.fromkeys(entry["lex_es"]))
        entry["xml_files_seen"] = sorted(entry["xml_files_seen"])
        out.append(entry)

    log(f"Canonical base entries after merging: {len(out)}")

    if collision_diagnostics:
        log(f"  WARNING: {len(collision_diagnostics)} base-key collisions had differing triples.")
        with open("entry_merge_collision_diagnostics_mixed_text_en_min2.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["base_id", "field", "kept", "seen", "xml_file"],
            )
            writer.writeheader()
            writer.writerows(collision_diagnostics)
        log("  Saved collision diagnostics → entry_merge_collision_diagnostics_mixed_text_en_min2.csv")

    return out


def filter_to_mixed_text_en_entries(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Keep only entries usable for the mixed-triples + English-text experiment.

    Requirements:
      - size >= 2
      - at least 2 EN mtriples
      - ES striples exist
      - EN and ES triplesets have the same number of triples
      - EN lexicalisation exists

    ES lexicalisation is not required or used.
    """
    kept = []
    dropped = []

    for entry in entries:
        reasons = []

        if int(entry["size"]) < MIN_TRIPLES:
            reasons.append(f"size_lt_{MIN_TRIPLES}")

        if len(entry["mtriples"]) < MIN_TRIPLES:
            reasons.append(f"n_mtriples_lt_{MIN_TRIPLES}")

        if not entry["striples"]:
            reasons.append("missing_es_striples")

        if entry["striples"] and len(entry["mtriples"]) != len(entry["striples"]):
            reasons.append("tripleset_triple_count_mismatch")

        if not entry["lex_en"]:
            reasons.append("missing_en_lex")

        if REQUIRE_PARALLEL_REPRESENTATIONS and reasons:
            row = {
                "base_id": make_entry_key_str(entry),
                "split": entry["split"],
                "category": entry["category"],
                "eid": entry["eid"],
                "size": entry["size"],
                "xml_file": entry["xml_file"],
                "reasons": ";".join(reasons),
                "n_mtriples": len(entry["mtriples"]),
                "n_striples": len(entry["striples"]),
                "n_lex_en": len(entry["lex_en"]),
                "n_lex_es": len(entry["lex_es"]),
            }
            dropped.append(row)
        else:
            kept.append(entry)

    if dropped:
        with open("dropped_non_mixed_text_en_entries_min2.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "base_id",
                    "split",
                    "category",
                    "eid",
                    "size",
                    "xml_file",
                    "reasons",
                    "n_mtriples",
                    "n_striples",
                    "n_lex_en",
                    "n_lex_es",
                ],
            )
            writer.writeheader()
            writer.writerows(dropped)

        log(f"  Dropped entries: {len(dropped)}")
        log("  Saved dropped entries → dropped_non_mixed_text_en_entries_min2.csv")

    log(f"Mixed-triples + English-text eligible base entries kept: {len(kept)}")
    return kept, dropped


# ─────────────────────────────────────────────────────────────
# QA MATCHING
# ─────────────────────────────────────────────────────────────

def qa_identity_key(qa: Dict) -> tuple:
    return (
        qa.get("split", ""),
        qa.get("category", ""),
        qa.get("eid", ""),
        qa.get("lex_id", ""),
        qa.get("xml_file", ""),
        qa.get("mtriple", ""),
        qa.get("question_idx", ""),
        qa.get("question_type", ""),
        qa.get("selected_role", ""),
        qa.get("question", ""),
        qa.get("question_es", ""),
    )


def get_matching_qa_rows(entry: Dict, qa_by_mtriple: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Relevance is still grounded in canonical EN mtriples from QA_CSV.
    """
    matched = []
    seen = set()

    for mt in entry["mtriples"]:
        norm_mt = normalise_triple(mt)

        for qa in qa_by_mtriple.get(norm_mt, []):
            key = qa_identity_key(qa)

            if key not in seen:
                seen.add(key)
                matched.append(qa)

    return matched


# ─────────────────────────────────────────────────────────────
# FIELD DEFINITIONS
# ─────────────────────────────────────────────────────────────

TRIPLE_DOC_FIELDS = [
    "doc_id",
    "base_id",
    "split",
    "category",
    "eid",
    "size",
    "xml_file",
    "lang",
    "triple_type",
    "n_triples",
    "triples",
    "supporting_mtriples",
    "mixed_lang_pattern",
    "mixed_n_en",
    "mixed_n_es",
    "mixed_strategy",
]

TEXT_DOC_FIELDS = [
    "doc_id",
    "base_id",
    "split",
    "category",
    "eid",
    "size",
    "xml_file",
    "lang",
    "lex_idx",
    "text",
    "supporting_mtriples",
    "text_selection_method",
]

REL_FIELDS = [
    "qa_category",
    "qa_eid",
    "qa_lex_id",
    "qa_mtriple",
    "qa_question",
    "qa_question_type",

    "qa_split",
    "qa_xml_file",
    "qa_question_idx",
    "qa_question_es",
    "qa_answer",
    "qa_answer_es",
    "qa_selected_role",
    "qa_validation_status",

    "doc_id",
    "relevance",
]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Load QA CSV ────────────────────────────────────────
    log("Loading QA CSV…")

    qa_rows = []

    with open(QA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "split",
            "category",
            "eid",
            "lex_id",
            "lex_text",
            "lex_text_es",
            "xml_file",
            "mtriple",
            "otriple",
            "statement",
            "question_idx",
            "question_type",
            "question",
            "answer",
            "question_es",
            "answer_es",
            "selected_role",
            "validation_status",
        }

        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise ValueError(
                f"QA_CSV is missing required columns: {sorted(missing_cols)}"
            )

        for row in reader:
            qa_rows.append(row)

    log(f"  QA rows: {len(qa_rows)}")

    qa_by_mtriple: Dict[str, List[Dict]] = defaultdict(list)

    for row in qa_rows:
        mt = row.get("mtriple", "").strip()

        if mt:
            qa_by_mtriple[normalise_triple(mt)].append(row)

    log(f"  Unique mtriples in QA: {len(qa_by_mtriple)}")

    # ── 2. Load + merge WebNLG entries ────────────────────────
    log("Loading full WebNLG dataset...")
    raw_entries = load_webnlg_all_sizes(WEBNLG_ROOT)

    log("Merging entries by canonical base key...")
    merged_entries = merge_entries_by_base_key(raw_entries)

    log(
        f"Filtering to size >= {MIN_TRIPLES} entries with aligned EN/ES triples "
        f"and English lexicalisation..."
    )
    all_entries, dropped_entries = filter_to_mixed_text_en_entries(merged_entries)

    # ── 3. Build documents & relations ───────────────────────
    log("Building mixed-language tripleset documents and English text documents...")

    triple_docs: List[Dict] = []
    text_docs: List[Dict] = []
    rel_triple: List[Dict] = []
    rel_text: List[Dict] = []

    seen_triple_ids: set = set()
    seen_text_ids: set = set()

    seen_rel_triple: set = set()
    seen_rel_text: set = set()

    mixed_diagnostics = []

    for entry in all_entries:
        base_id = make_entry_key_str(entry)
        mtriples = entry["mtriples"]

        mixed_triples, mixed_pattern = make_mixed_tripleset(entry)

        mixed_n_en = mixed_pattern.count("en")
        mixed_n_es = mixed_pattern.count("es")

        mixed_diagnostics.append({
            "base_id": base_id,
            "split": entry["split"],
            "category": entry["category"],
            "eid": entry["eid"],
            "size": entry["size"],
            "n_triples": len(mixed_triples),
            "mixed_n_en": mixed_n_en,
            "mixed_n_es": mixed_n_es,
            "mixed_lang_pattern": " ".join(mixed_pattern),
            "mixed_triples": TRIPLE_SEP.join(mixed_triples),
            "mtriples": TRIPLE_SEP.join(entry["mtriples"]),
            "striples": TRIPLE_SEP.join(entry["striples"]),
        })

        matching_qa_rows = get_matching_qa_rows(entry, qa_by_mtriple)

        # ── 3a. Mixed tripleset document ──────────────────────
        # Only one tripleset document is written, labelled as EN.
        triple_doc_id = make_tripleset_doc_id(base_id, DOC_LANG_LABEL)

        if triple_doc_id in seen_triple_ids:
            raise ValueError(f"Duplicate tripleset doc_id detected: {triple_doc_id}")

        seen_triple_ids.add(triple_doc_id)

        triple_docs.append({
            "doc_id": triple_doc_id,
            "base_id": base_id,
            "split": entry["split"],
            "category": entry["category"],
            "eid": entry["eid"],
            "size": entry["size"],
            "xml_file": entry["xml_file"],
            "lang": DOC_LANG_LABEL,
            "triple_type": "mixedtripleset_50_50_label_en",
            "n_triples": len(mixed_triples),
            "triples": TRIPLE_SEP.join(mixed_triples),
            "supporting_mtriples": TRIPLE_SEP.join(mtriples),
            "mixed_lang_pattern": " ".join(mixed_pattern),
            "mixed_n_en": mixed_n_en,
            "mixed_n_es": mixed_n_es,
            "mixed_strategy": "random_50_50_aligned_by_position_label_en",
        })

        for qa in matching_qa_rows:
            rel_key = (triple_doc_id,) + qa_identity_key(qa)

            if rel_key not in seen_rel_triple:
                seen_rel_triple.add(rel_key)

                rel_triple.append({
                    "qa_category": qa.get("category", ""),
                    "qa_eid": qa.get("eid", ""),
                    "qa_lex_id": qa.get("lex_id", ""),
                    "qa_mtriple": qa.get("mtriple", ""),
                    "qa_question": qa.get("question", ""),
                    "qa_question_type": qa.get("question_type", ""),

                    "qa_split": qa.get("split", ""),
                    "qa_xml_file": qa.get("xml_file", ""),
                    "qa_question_idx": qa.get("question_idx", ""),
                    "qa_question_es": qa.get("question_es", ""),
                    "qa_answer": qa.get("answer", ""),
                    "qa_answer_es": qa.get("answer_es", ""),
                    "qa_selected_role": qa.get("selected_role", ""),
                    "qa_validation_status": qa.get("validation_status", ""),

                    "doc_id": triple_doc_id,
                    "relevance": 1,
                })

        # ── 3b. English text document ─────────────────────────
        # Only one text document is written, labelled as EN.
        choice_key = f"{base_id}::{DOC_LANG_LABEL}"
        chosen_idx, chosen_lex = stable_choice(entry["lex_en"], choice_key)
        text_doc_id = make_text_doc_id(base_id, DOC_LANG_LABEL)

        if text_doc_id in seen_text_ids:
            raise ValueError(f"Duplicate text doc_id detected: {text_doc_id}")

        seen_text_ids.add(text_doc_id)

        text_docs.append({
            "doc_id": text_doc_id,
            "base_id": base_id,
            "split": entry["split"],
            "category": entry["category"],
            "eid": entry["eid"],
            "size": entry["size"],
            "xml_file": entry["xml_file"],
            "lang": DOC_LANG_LABEL,
            "lex_idx": chosen_idx,
            "text": chosen_lex,
            "supporting_mtriples": TRIPLE_SEP.join(mtriples),
            "text_selection_method": "stable_choice_english_only",
        })

        for qa in matching_qa_rows:
            rel_key = (text_doc_id,) + qa_identity_key(qa)

            if rel_key not in seen_rel_text:
                seen_rel_text.add(rel_key)

                rel_text.append({
                    "qa_category": qa.get("category", ""),
                    "qa_eid": qa.get("eid", ""),
                    "qa_lex_id": qa.get("lex_id", ""),
                    "qa_mtriple": qa.get("mtriple", ""),
                    "qa_question": qa.get("question", ""),
                    "qa_question_type": qa.get("question_type", ""),

                    "qa_split": qa.get("split", ""),
                    "qa_xml_file": qa.get("xml_file", ""),
                    "qa_question_idx": qa.get("question_idx", ""),
                    "qa_question_es": qa.get("question_es", ""),
                    "qa_answer": qa.get("answer", ""),
                    "qa_answer_es": qa.get("answer_es", ""),
                    "qa_selected_role": qa.get("selected_role", ""),
                    "qa_validation_status": qa.get("validation_status", ""),

                    "doc_id": text_doc_id,
                    "relevance": 1,
                })

    log(f"  Tripleset documents : {len(triple_docs)}")
    log(f"  Text documents      : {len(text_docs)}")
    log(f"  Tripleset relations : {len(rel_triple)}")
    log(f"  Text relations      : {len(rel_text)}")

    # ── 4. Strict checks before writing ───────────────────────
    log("Running strict one-language inventory checks...")

    triple_base_ids = {d["base_id"] for d in triple_docs}
    text_base_ids = {d["base_id"] for d in text_docs}

    if triple_base_ids != text_base_ids:
        raise ValueError(
            "Non-parallel base IDs between mixed triples and English text: "
            f"tripleset_only={len(triple_base_ids - text_base_ids)}, "
            f"text_only={len(text_base_ids - triple_base_ids)}"
        )

    if any(d["lang"] != DOC_LANG_LABEL for d in triple_docs):
        raise ValueError("Found tripleset docs with a language label other than DOC_LANG_LABEL.")

    if any(d["lang"] != DOC_LANG_LABEL for d in text_docs):
        raise ValueError("Found text docs with a language label other than DOC_LANG_LABEL.")

    triple_counts = Counter(d["size"] for d in triple_docs)
    text_counts = Counter(d["size"] for d in text_docs)

    if triple_counts != text_counts:
        raise ValueError(
            "Tripleset/text size distributions differ:\n"
            f"triples={dict(triple_counts)}\n"
            f"texts={dict(text_counts)}"
        )

    log("  Strict checks passed.")

    # ── 5. Write CSVs ─────────────────────────────────────────
    log("Writing output files...")

    write_csv(OUT_TRIPLESETS, TRIPLE_DOC_FIELDS, triple_docs)
    write_csv(OUT_TEXTS, TEXT_DOC_FIELDS, text_docs)
    write_csv(OUT_REL_TRIPLE, REL_FIELDS, rel_triple)
    write_csv(OUT_REL_TEXT, REL_FIELDS, rel_text)

    with open("mixed_tripleset_text_en_diagnostics_min2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_id",
                "split",
                "category",
                "eid",
                "size",
                "n_triples",
                "mixed_n_en",
                "mixed_n_es",
                "mixed_lang_pattern",
                "mixed_triples",
                "mtriples",
                "striples",
            ],
        )
        writer.writeheader()
        writer.writerows(mixed_diagnostics)

    log("  Saved mixed triples diagnostics → mixed_tripleset_text_en_diagnostics_min2.csv")

    # ── 6. Coverage + filtered QA ─────────────────────────────
    covered_mtriples_triple = {normalise_triple(r["qa_mtriple"]) for r in rel_triple}
    covered_mtriples_text = {normalise_triple(r["qa_mtriple"]) for r in rel_text}
    covered_mtriples_both = covered_mtriples_triple & covered_mtriples_text

    all_qa_mtriples = set(qa_by_mtriple.keys())
    excluded_from_full_qa = sorted(all_qa_mtriples - covered_mtriples_both)

    qa_rows_filtered = write_filtered_qa_csv(
        OUT_QA,
        qa_rows,
        covered_mtriples_both,
    )

    qa_mtriples_filtered = {
        normalise_triple(row.get("mtriple", ""))
        for row in qa_rows_filtered
        if row.get("mtriple", "").strip()
    }

    if excluded_from_full_qa:
        with open("uncovered_mtriples_excluded_from_mixed_text_en_min2.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "mtriple_normalized",
                    "n_qa_rows",
                    "example_question",
                    "example_question_es",
                ],
            )
            writer.writeheader()

            for mt in excluded_from_full_qa:
                rows = qa_by_mtriple.get(mt, [])
                writer.writerow({
                    "mtriple_normalized": mt,
                    "n_qa_rows": len(rows),
                    "example_question": rows[0].get("question", "") if rows else "",
                    "example_question_es": rows[0].get("question_es", "") if rows else "",
                })

        log("  Saved excluded mtriples → uncovered_mtriples_excluded_from_mixed_text_en_min2.csv")

    # ── 7. Summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("IR DOCUMENT COLLECTION SUMMARY — MIXED TRIPLES + ENGLISH TEXT MIN2")
    print("=" * 80)

    print("\nBase-entry inventory:")
    print(f"  raw XML entries loaded          : {len(raw_entries):,}")
    print(f"  merged canonical base entries   : {len(merged_entries):,}")
    print(f"  eligible entries kept           : {len(all_entries):,}")
    print(f"  entries dropped                 : {len(dropped_entries):,}")

    print("\nDocument counts:")
    print(f"  mixed tripleset documents : {len(triple_docs):,}")
    print(f"  English text documents    : {len(text_docs):,}")

    print("\nDocument language label:")
    print(f"  mixed triples lang : {DOC_LANG_LABEL}")
    print(f"  text lang          : {DOC_LANG_LABEL}")

    print("\nSize distribution:")
    all_size_keys = sorted(set(triple_counts) | set(text_counts))
    for size in all_size_keys:
        print(
            f"  size={size}: "
            f"mixed_triplesets={triple_counts[size]:,}, "
            f"english_texts={text_counts[size]:,}"
        )

    print("\nMixed-language pattern summary:")
    mixed_pattern_counts = Counter(d["mixed_lang_pattern"] for d in triple_docs)
    print(f"  unique mixed patterns: {len(mixed_pattern_counts):,}")
    for pattern, count in mixed_pattern_counts.most_common(10):
        print(f"  {pattern}: {count:,}")

    mixed_size_lang_counts = Counter(
        (d["size"], d["mixed_n_en"], d["mixed_n_es"])
        for d in triple_docs
    )

    print("\nMixed EN/ES triple counts by size:")
    for key, count in sorted(mixed_size_lang_counts.items()):
        size, n_en, n_es = key
        print(f"  size={size}, EN={n_en}, ES={n_es}: {count:,}")

    print("\nDeduplication check:")
    triple_id_list = [d["doc_id"] for d in triple_docs]
    text_id_list = [d["doc_id"] for d in text_docs]

    dup_t = len(triple_id_list) - len(set(triple_id_list))
    dup_x = len(text_id_list) - len(set(text_id_list))

    print(f"  Tripleset doc duplicates : {dup_t} {'✅' if dup_t == 0 else '⚠'}")
    print(f"  Text doc duplicates      : {dup_x} {'✅' if dup_x == 0 else '⚠'}")

    rt_keys = [
        (
            r["doc_id"],
            r["qa_split"],
            r["qa_category"],
            r["qa_eid"],
            r["qa_lex_id"],
            r["qa_xml_file"],
            r["qa_mtriple"],
            r["qa_question_idx"],
            r["qa_question_type"],
            r["qa_selected_role"],
            r["qa_question"],
            r["qa_question_es"],
        )
        for r in rel_triple
    ]

    rx_keys = [
        (
            r["doc_id"],
            r["qa_split"],
            r["qa_category"],
            r["qa_eid"],
            r["qa_lex_id"],
            r["qa_xml_file"],
            r["qa_mtriple"],
            r["qa_question_idx"],
            r["qa_question_type"],
            r["qa_selected_role"],
            r["qa_question"],
            r["qa_question_es"],
        )
        for r in rel_text
    ]

    dup_rt = len(rt_keys) - len(set(rt_keys))
    dup_rx = len(rx_keys) - len(set(rx_keys))

    print(f"  Tripleset rel duplicates : {dup_rt} {'✅' if dup_rt == 0 else '⚠'}")
    print(f"  Text rel duplicates      : {dup_rx} {'✅' if dup_rx == 0 else '⚠'}")

    print("\nQA mtriple coverage relative to FULL QA CSV:")
    print(f"  covered by mixed triples : {len(covered_mtriples_triple):,}/{len(all_qa_mtriples):,}")
    print(f"  covered by English text  : {len(covered_mtriples_text):,}/{len(all_qa_mtriples):,}")
    print(f"  covered by both          : {len(covered_mtriples_both):,}/{len(all_qa_mtriples):,}")
    print(f"  excluded from experiment : {len(excluded_from_full_qa):,}")

    print("\nQA mtriple coverage relative to FILTERED QA CSV:")
    print(f"  covered by mixed triples : {len(covered_mtriples_triple & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")
    print(f"  covered by English text  : {len(covered_mtriples_text & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")
    print(f"  covered by both          : {len(covered_mtriples_both & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")

    print("\nOutput files:")
    print(f"  Filtered QA:       {OUT_QA}")
    print(f"  Mixed triplesets:  {OUT_TRIPLESETS}")
    print(f"  English texts:     {OUT_TEXTS}")
    print(f"  Tripleset qrels:   {OUT_REL_TRIPLE}")
    print(f"  Text qrels:        {OUT_REL_TEXT}")

    print("=" * 80)