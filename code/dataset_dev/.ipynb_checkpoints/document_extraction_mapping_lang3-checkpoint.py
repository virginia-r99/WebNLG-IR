"""
Create a single aligned WebNLG IR dataset with three document-language conditions:

  lang=en   -> English/canonical triples + English lexicalisation
  lang=es   -> Spanish triples + Spanish lexicalisation
  lang=mix  -> mixed EN/ES triples + English lexicalisation

This script merges the previous monolingual and mixed extraction logic into one
controlled mapping. It is intended for experiments where mixed-language KG
triplesets are compared against monolingual EN/ES triplesets using the same
base-entry inventory, the same QA rows, the same qrels, and the same text source.

Recommended preparation call after running this script:

python prepare_webnlg_ir.py \
  --qa_csv './dataset_lang3_min2/webnlg_qa_selected_es_v4_validated_lang3_min2.csv' \
  --triplesets_csv './dataset_lang3_min2/ir_triplesets_lang3_min2.csv' \
  --texts_csv './dataset_lang3_min2/ir_texts_lang3_min2.csv' \
  --out_dir ./prepared_lang3_min2_normtext \
  --min_triples 1 \
  --triple_variant normalized \
  --text_variant normalized

Use --min_triples 1 at preparation time because this script already restricts
entries to size >= MIN_TRIPLES.
"""

import argparse
import csv
import glob
import hashlib
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────

DEFAULT_WEBNLG_ROOT = "./WebNLG_ES"
DEFAULT_QA_CSV = "webnlg_qa_selected_es_v4_validated.csv"
DEFAULT_OUT_DIR = "dataset_lang3_min2"

RANDOM_SEED = 42
TRIPLE_SEP = " ||| "
MIN_TRIPLES = 1

LANGS = ["en", "es", "mix"]


# ─────────────────────────────────────────────────────────────
# Logging / IO
# ─────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[LOG] {msg}", flush=True)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"  Saved {len(rows):>7} rows -> {path}")


def write_filtered_qa_csv(path: Path, original_qa_rows: List[Dict], covered_mtriples: set) -> List[Dict]:
    """
    Keep only QA rows whose canonical mtriple is covered by the final aligned
    lang3 document inventory.
    """
    if not original_qa_rows:
        raise ValueError("No QA rows to write.")

    fieldnames = list(original_qa_rows[0].keys())

    kept_rows = [
        row
        for row in original_qa_rows
        if normalise_triple(row.get("mtriple", "")) in covered_mtriples
    ]

    write_csv(path, fieldnames, kept_rows)
    return kept_rows


# ─────────────────────────────────────────────────────────────
# Normalization / IDs
# ─────────────────────────────────────────────────────────────

def normalise_triple(t: str) -> str:
    return " ".join(str(t).lower().strip().split())


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def make_entry_key_tuple(entry: Dict) -> Tuple[str, str, str, int]:
    """
    Canonical base-entry key.

    We intentionally do not include lex_idx, because the retrieval unit is the
    WebNLG entry/tripleset, not an individual lexicalisation variant.
    """
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


def make_tripleset_doc_id(base_id: str, lang: str) -> str:
    return f"tripleset__{base_id}__{lang}"


def make_text_doc_id(base_id: str, lang: str) -> str:
    return f"text__{base_id}__{lang}"


def stable_choice(items: List[str], key: str, random_seed: int) -> Tuple[int, str]:
    """
    Deterministic pseudo-random choice, stable across runs and machines.
    """
    if not items:
        raise ValueError("stable_choice received an empty list.")

    digest = hashlib.md5(f"{random_seed}::{key}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(items)
    return idx, items[idx]


def stable_rng(key: str, random_seed: int) -> random.Random:
    digest = hashlib.md5(f"{random_seed}::{key}".encode("utf-8")).hexdigest()
    seed = int(digest, 16) % (2**32)
    return random.Random(seed)


# ─────────────────────────────────────────────────────────────
# Mixed tripleset construction
# ─────────────────────────────────────────────────────────────

def make_mixed_tripleset(entry: Dict, min_triples: int, random_seed: int) -> Tuple[List[str], List[str]]:
    """
    Build a mixed-language tripleset from aligned EN/ES triplesets.

    Assumption:
      The English and Spanish triplesets are aligned by position:
      mtriples[i] and striples[i] are alternative language realisations of the
      same triple/fact.

    This preserves the original tripleset meaning:
      - no triples are moved across entries
      - no facts are shuffled
      - original triple order is preserved
      - only the language of each triple position is selected

    For n triples:
      - if n is even, use n/2 EN and n/2 ES.
      - if n is odd, use floor(n/2) from one language and ceil(n/2) from the other.
        The language receiving the extra triple is chosen deterministically at
        random per entry.
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

    if n < min_triples:
        raise ValueError(f"Cannot mix entry {base_id}: tripleset size {n} < {min_triples}")

    rng = stable_rng(f"mix::{base_id}", random_seed)

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
# Parsing WebNLG XML
# ─────────────────────────────────────────────────────────────

def parse_webnlg_xml_full(xml_path: str, split: str) -> List[Dict]:
    """
    Parses one WebNLG_ES XML file.

    Extracts:
      - EN/canonical modified triples: <modifiedtripleset><mtriple>...</mtriple>
      - ES triples: <spanishtripleset><striple>...</striple>
      - EN lexicalisations: <lex lang="en">
      - ES lexicalisations: <lex lang="es">

    Relevance is grounded through QA_CSV.mtriple, which corresponds to canonical
    modified triples.
    """
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

        # Canonical / English modified triples.
        mtriples = [
            mt.text.strip()
            for mt in entry.findall("./modifiedtripleset/mtriple")
            if mt.text and mt.text.strip()
        ]

        # Spanish triples.
        striples = [
            st.text.strip()
            for st in entry.findall("./spanishtripleset/striple")
            if st.text and st.text.strip()
        ]

        # Fallbacks for non-standard nesting.
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

        # Without canonical mtriples, the entry cannot be connected to QA rows.
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
# Merge / filter entries
# ─────────────────────────────────────────────────────────────

def merge_entries_by_base_key(entries: List[Dict], out_dir: Path) -> List[Dict]:
    """
    Merge duplicated raw XML entries that map to the same base entry.

    Tripleset docs and text docs are created from exactly the same canonical
    base-entry inventory.
    """
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

        # Keep the first canonical tripleset if it differs, but log it.
        if list(entry["mtriples"]) != list(current["mtriples"]):
            collision_diagnostics.append({
                "base_id": make_entry_key_str_from_tuple(key),
                "field": "mtriples",
                "kept": TRIPLE_SEP.join(current["mtriples"]),
                "seen": TRIPLE_SEP.join(entry["mtriples"]),
                "xml_file": entry["xml_file"],
            })

        # Keep the first Spanish tripleset if it differs, but log it.
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

        # Merge lexicalisations and deduplicate text strings.
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
        path = out_dir / "entry_merge_collision_diagnostics_lang3_min2.csv"
        write_csv(
            path,
            ["base_id", "field", "kept", "seen", "xml_file"],
            collision_diagnostics,
        )

    return out


def filter_to_lang3_entries(entries: List[Dict], min_triples: int, out_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Keep only entries usable for the controlled EN/ES/MIX dataset.

    Requirements:
      - size >= min_triples
      - at least min_triples EN mtriples
      - ES striples exist
      - EN and ES triplesets have the same number of triples
      - EN lexicalisation exists
      - ES lexicalisation exists

    The ES lexicalisation is required so that en/es/mix are produced from the
    same inventory as the original parallel mapping. For lang=mix, text is the
    English lexicalisation by design.
    """
    kept = []
    dropped = []

    for entry in entries:
        reasons = []

        if int(entry["size"]) < min_triples:
            reasons.append(f"size_lt_{min_triples}")

        if len(entry["mtriples"]) < min_triples:
            reasons.append(f"n_mtriples_lt_{min_triples}")

        if not entry["striples"]:
            reasons.append("missing_es_striples")

        if entry["striples"] and len(entry["mtriples"]) != len(entry["striples"]):
            reasons.append("tripleset_triple_count_mismatch")

        if not entry["lex_en"]:
            reasons.append("missing_en_lex")

        if not entry["lex_es"]:
            reasons.append("missing_es_lex")

        if reasons:
            dropped.append({
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
            })
        else:
            kept.append(entry)

    if dropped:
        path = out_dir / "dropped_non_lang3_entries_min2.csv"
        write_csv(
            path,
            [
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
            dropped,
        )

    log(f"Lang3 eligible base entries kept: {len(kept)}")
    return kept, dropped


# ─────────────────────────────────────────────────────────────
# QA matching
# ─────────────────────────────────────────────────────────────

def qa_identity_key(qa: Dict) -> tuple:
    """
    Stable identity for one QA row.
    """
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
    Relevance is determined by canonical mtriples from QA_CSV.

    If a QA mtriple appears in an entry's canonical mtriples, then all lang3
    docs for that base entry are relevant:
      - tripleset en / es / mix
      - text en / es / mix
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
# Field definitions
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
    "text_source_lang",
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
# Main dataset builder
# ─────────────────────────────────────────────────────────────

def load_qa_rows(qa_csv: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    log("Loading QA CSV...")

    qa_rows = []

    with open(qa_csv, newline="", encoding="utf-8") as f:
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
            raise ValueError(f"QA_CSV is missing required columns: {sorted(missing_cols)}")

        for row in reader:
            qa_rows.append(row)

    log(f"  QA rows: {len(qa_rows)}")

    qa_by_mtriple: Dict[str, List[Dict]] = defaultdict(list)

    for row in qa_rows:
        mt = row.get("mtriple", "").strip()

        if mt:
            qa_by_mtriple[normalise_triple(mt)].append(row)

    log(f"  Unique mtriples in QA: {len(qa_by_mtriple)}")

    return qa_rows, qa_by_mtriple


def build_lang3_dataset(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_seed)

    out_qa = out_dir / f"webnlg_qa_selected_es_v4_validated_lang3_min{args.min_triples}.csv"
    out_triplesets = out_dir / f"ir_triplesets_lang3_min{args.min_triples}.csv"
    out_texts = out_dir / f"ir_texts_lang3_min{args.min_triples}.csv"
    out_rel_triples = out_dir / f"ir_relations_triplesets_lang3_min{args.min_triples}.csv"
    out_rel_texts = out_dir / f"ir_relations_texts_lang3_min{args.min_triples}.csv"

    qa_rows, qa_by_mtriple = load_qa_rows(args.qa_csv)

    log("Loading full WebNLG dataset...")
    raw_entries = load_webnlg_all_sizes(args.webnlg_root)

    log("Merging entries by canonical base key...")
    merged_entries = merge_entries_by_base_key(raw_entries, out_dir)

    log(
        f"Filtering to size >= {args.min_triples}, aligned EN/ES triplesets, "
        "and EN/ES lexicalisations..."
    )
    all_entries, dropped_entries = filter_to_lang3_entries(
        merged_entries,
        min_triples=args.min_triples,
        out_dir=out_dir,
    )

    log("Building lang3 tripleset/text documents and relations...")

    triple_docs: List[Dict] = []
    text_docs: List[Dict] = []
    rel_triple: List[Dict] = []
    rel_text: List[Dict] = []

    seen_triple_ids = set()
    seen_text_ids = set()
    seen_rel_triple = set()
    seen_rel_text = set()

    mixed_diagnostics = []
    eligible_base_rows = []

    for entry in all_entries:
        base_id = make_entry_key_str(entry)
        mtriples = entry["mtriples"]
        striples = entry["striples"]

        mixed_triples, mixed_pattern = make_mixed_tripleset(
            entry,
            min_triples=args.min_triples,
            random_seed=args.random_seed,
        )

        mixed_n_en = mixed_pattern.count("en")
        mixed_n_es = mixed_pattern.count("es")

        choice_key_en = f"{base_id}::en"
        chosen_en_idx, chosen_en_lex = stable_choice(entry["lex_en"], choice_key_en, args.random_seed)

        choice_key_es = f"{base_id}::es"
        chosen_es_idx, chosen_es_lex = stable_choice(entry["lex_es"], choice_key_es, args.random_seed)

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
            "mtriples": TRIPLE_SEP.join(mtriples),
            "striples": TRIPLE_SEP.join(striples),
        })

        eligible_base_rows.append({
            "base_id": base_id,
            "split": entry["split"],
            "category": entry["category"],
            "eid": entry["eid"],
            "size": entry["size"],
            "n_mtriples": len(mtriples),
            "n_striples": len(striples),
            "n_lex_en": len(entry["lex_en"]),
            "n_lex_es": len(entry["lex_es"]),
        })

        matching_qa_rows = get_matching_qa_rows(entry, qa_by_mtriple)

        # ── Tripleset documents: en, es, mix ─────────────────
        tripleset_variants = [
            {
                "lang": "en",
                "triple_type": "modifiedtripleset",
                "triples": mtriples,
                "mixed_lang_pattern": "",
                "mixed_n_en": "",
                "mixed_n_es": "",
                "mixed_strategy": "",
            },
            {
                "lang": "es",
                "triple_type": "spanishtripleset",
                "triples": striples,
                "mixed_lang_pattern": "",
                "mixed_n_en": "",
                "mixed_n_es": "",
                "mixed_strategy": "",
            },
            {
                "lang": "mix",
                "triple_type": "mixedtripleset_50_50",
                "triples": mixed_triples,
                "mixed_lang_pattern": " ".join(mixed_pattern),
                "mixed_n_en": mixed_n_en,
                "mixed_n_es": mixed_n_es,
                "mixed_strategy": "random_50_50_aligned_by_position_text_mix_is_en",
            },
        ]

        for variant in tripleset_variants:
            lang = variant["lang"]
            triples = variant["triples"]
            triple_doc_id = make_tripleset_doc_id(base_id, lang)

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
                "lang": lang,
                "triple_type": variant["triple_type"],
                "n_triples": len(triples),
                "triples": TRIPLE_SEP.join(triples),
                "supporting_mtriples": TRIPLE_SEP.join(mtriples),
                "mixed_lang_pattern": variant["mixed_lang_pattern"],
                "mixed_n_en": variant["mixed_n_en"],
                "mixed_n_es": variant["mixed_n_es"],
                "mixed_strategy": variant["mixed_strategy"],
            })

            for qa in matching_qa_rows:
                rel_key = (triple_doc_id,) + qa_identity_key(qa)

                if rel_key in seen_rel_triple:
                    continue

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

        # ── Text documents: en, es, mix ───────────────────────
        # text_mix deliberately uses the English lexicalisation.
        text_variants = [
            {
                "lang": "en",
                "lex_idx": chosen_en_idx,
                "text": chosen_en_lex,
                "text_source_lang": "en",
                "text_selection_method": "stable_choice_en",
            },
            {
                "lang": "es",
                "lex_idx": chosen_es_idx,
                "text": chosen_es_lex,
                "text_source_lang": "es",
                "text_selection_method": "stable_choice_es",
            },
            {
                "lang": "mix",
                "lex_idx": chosen_en_idx,
                "text": chosen_en_lex,
                "text_source_lang": "en",
                "text_selection_method": "stable_choice_en_for_mix",
            },
        ]

        for variant in text_variants:
            lang = variant["lang"]
            text_doc_id = make_text_doc_id(base_id, lang)

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
                "lang": lang,
                "lex_idx": variant["lex_idx"],
                "text": variant["text"],
                "supporting_mtriples": TRIPLE_SEP.join(mtriples),
                "text_source_lang": variant["text_source_lang"],
                "text_selection_method": variant["text_selection_method"],
            })

            for qa in matching_qa_rows:
                rel_key = (text_doc_id,) + qa_identity_key(qa)

                if rel_key in seen_rel_text:
                    continue

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

    # ── Strict checks ────────────────────────────────────────
    log("Running strict lang3 inventory checks...")

    triple_base_by_lang = defaultdict(set)
    text_base_by_lang = defaultdict(set)

    for d in triple_docs:
        triple_base_by_lang[d["lang"]].add(d["base_id"])

    for d in text_docs:
        text_base_by_lang[d["lang"]].add(d["base_id"])

    expected_langs = set(LANGS)

    if set(triple_base_by_lang) != expected_langs:
        raise ValueError(f"Tripleset languages differ from expected: {set(triple_base_by_lang)}")

    if set(text_base_by_lang) != expected_langs:
        raise ValueError(f"Text languages differ from expected: {set(text_base_by_lang)}")

    for lang in LANGS:
        tset = triple_base_by_lang[lang]
        xset = text_base_by_lang[lang]

        if tset != xset:
            raise ValueError(
                f"Non-parallel base IDs for lang={lang}: "
                f"tripleset_only={len(tset - xset)}, text_only={len(xset - tset)}"
            )

    if not (
        triple_base_by_lang["en"]
        == triple_base_by_lang["es"]
        == triple_base_by_lang["mix"]
    ):
        raise ValueError("Tripleset base inventories differ across en/es/mix.")

    if not (
        text_base_by_lang["en"]
        == text_base_by_lang["es"]
        == text_base_by_lang["mix"]
    ):
        raise ValueError("Text base inventories differ across en/es/mix.")

    triple_counts = Counter((d["lang"], d["size"]) for d in triple_docs)
    text_counts = Counter((d["lang"], d["size"]) for d in text_docs)

    if triple_counts != text_counts:
        raise ValueError(
            "Tripleset/text size distributions differ:\n"
            f"triples={dict(triple_counts)}\n"
            f"texts={dict(text_counts)}"
        )

    # text_mix must be identical to text_en for every base_id.
    text_by_base_lang = {(d["base_id"], d["lang"]): d["text"] for d in text_docs}
    for base_id in text_base_by_lang["en"]:
        if text_by_base_lang[(base_id, "en")] != text_by_base_lang[(base_id, "mix")]:
            raise ValueError(f"text_mix differs from text_en for base_id={base_id}")

    log("  Strict lang3 checks passed.")

    # ── Write outputs ────────────────────────────────────────
    log("Writing output files...")

    write_csv(out_triplesets, TRIPLE_DOC_FIELDS, triple_docs)
    write_csv(out_texts, TEXT_DOC_FIELDS, text_docs)
    write_csv(out_rel_triples, REL_FIELDS, rel_triple)
    write_csv(out_rel_texts, REL_FIELDS, rel_text)

    write_csv(
        out_dir / f"mixed_tripleset_diagnostics_lang3_min{args.min_triples}.csv",
        [
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
        mixed_diagnostics,
    )

    write_csv(
        out_dir / f"eligible_base_ids_lang3_min{args.min_triples}.csv",
        [
            "base_id",
            "split",
            "category",
            "eid",
            "size",
            "n_mtriples",
            "n_striples",
            "n_lex_en",
            "n_lex_es",
        ],
        eligible_base_rows,
    )

    # ── Filtered QA ──────────────────────────────────────────
    covered_mtriples_triple = {normalise_triple(r["qa_mtriple"]) for r in rel_triple}
    covered_mtriples_text = {normalise_triple(r["qa_mtriple"]) for r in rel_text}
    covered_mtriples_both = covered_mtriples_triple & covered_mtriples_text

    qa_rows_filtered = write_filtered_qa_csv(out_qa, qa_rows, covered_mtriples_both)

    qa_mtriples_filtered = {
        normalise_triple(row.get("mtriple", ""))
        for row in qa_rows_filtered
        if row.get("mtriple", "").strip()
    }

    all_qa_mtriples = set(qa_by_mtriple.keys())
    excluded_from_full_qa = sorted(all_qa_mtriples - covered_mtriples_both)

    if excluded_from_full_qa:
        excluded_rows = []
        for mt in excluded_from_full_qa:
            rows = qa_by_mtriple.get(mt, [])
            excluded_rows.append({
                "mtriple_normalized": mt,
                "n_qa_rows": len(rows),
                "example_question": rows[0].get("question", "") if rows else "",
                "example_question_es": rows[0].get("question_es", "") if rows else "",
            })

        write_csv(
            out_dir / f"uncovered_mtriples_excluded_from_lang3_min{args.min_triples}.csv",
            ["mtriple_normalized", "n_qa_rows", "example_question", "example_question_es"],
            excluded_rows,
        )

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"IR DOCUMENT COLLECTION SUMMARY — LANG3 MIN{args.min_triples}")
    print("=" * 80)

    print("\nBase-entry inventory:")
    print(f"  raw XML entries loaded        : {len(raw_entries):,}")
    print(f"  merged canonical base entries : {len(merged_entries):,}")
    print(f"  lang3 eligible entries kept   : {len(all_entries):,}")
    print(f"  entries dropped               : {len(dropped_entries):,}")

    print("\nDocument counts:")
    print(f"  tripleset documents : {len(triple_docs):,}")
    print(f"  text documents      : {len(text_docs):,}")

    print("\nTripleset document language distribution:")
    for lang, count in sorted(Counter(d["lang"] for d in triple_docs).items()):
        print(f"  lang={lang} : {count:,}")

    print("\nText document language distribution:")
    for lang, count in sorted(Counter(d["lang"] for d in text_docs).items()):
        print(f"  lang={lang} : {count:,}")

    print("\nText source language distribution:")
    for source_lang, count in sorted(Counter(d["text_source_lang"] for d in text_docs).items()):
        print(f"  text_source_lang={source_lang} : {count:,}")

    print("\nTripleset document type distribution:")
    for typ, count in sorted(Counter(d["triple_type"] for d in triple_docs).items()):
        print(f"  {typ} : {count:,}")

    print("\nSize distribution by language:")
    all_size_keys = sorted(set(triple_counts) | set(text_counts))
    for lang, size in all_size_keys:
        print(
            f"  lang={lang}, size={size}: "
            f"triplesets={triple_counts[(lang, size)]:,}, "
            f"texts={text_counts[(lang, size)]:,}"
        )

    print("\nMixed-language pattern summary:")
    mixed_pattern_counts = Counter(d["mixed_lang_pattern"] for d in triple_docs if d["lang"] == "mix")
    print(f"  unique mixed patterns: {len(mixed_pattern_counts):,}")
    for pattern, count in mixed_pattern_counts.most_common(10):
        print(f"  {pattern}: {count:,}")

    mixed_size_lang_counts = Counter(
        (d["size"], d["mixed_n_en"], d["mixed_n_es"])
        for d in triple_docs
        if d["lang"] == "mix"
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

    print(f"  Tripleset doc duplicates : {dup_t}")
    print(f"  Text doc duplicates      : {dup_x}")

    print("\nQA mtriple coverage relative to FULL QA CSV:")
    print(f"  covered by triplesets : {len(covered_mtriples_triple):,}/{len(all_qa_mtriples):,}")
    print(f"  covered by texts      : {len(covered_mtriples_text):,}/{len(all_qa_mtriples):,}")
    print(f"  covered by both       : {len(covered_mtriples_both):,}/{len(all_qa_mtriples):,}")
    print(f"  excluded from lang3   : {len(excluded_from_full_qa):,}")

    print("\nQA mtriple coverage relative to FILTERED QA CSV:")
    print(f"  covered by triplesets : {len(covered_mtriples_triple & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")
    print(f"  covered by texts      : {len(covered_mtriples_text & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")
    print(f"  covered by both       : {len(covered_mtriples_both & qa_mtriples_filtered):,}/{len(qa_mtriples_filtered):,}")

    print("\nOutput files:")
    print(f"  Filtered QA:       {out_qa}")
    print(f"  Triplesets:        {out_triplesets}")
    print(f"  Texts:             {out_texts}")
    print(f"  Tripleset qrels:   {out_rel_triples}")
    print(f"  Text qrels:        {out_rel_texts}")

    print("=" * 80)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Create a controlled en/es/mix WebNLG IR mapping dataset."
    )

    p.add_argument("--webnlg_root", default=DEFAULT_WEBNLG_ROOT)
    p.add_argument("--qa_csv", default=DEFAULT_QA_CSV)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--min_triples", type=int, default=MIN_TRIPLES)
    p.add_argument("--random_seed", type=int, default=RANDOM_SEED)

    return p.parse_args()


if __name__ == "__main__":
    build_lang3_dataset(parse_args())
