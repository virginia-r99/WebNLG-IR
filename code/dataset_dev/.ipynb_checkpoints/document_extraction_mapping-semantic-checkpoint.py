import os
import csv
import glob
import hashlib
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

WEBNLG_ROOT = "./WebNLG_ES"

# Expected QA columns:
# split,category,eid,lex_id,lex_text,lex_text_es,xml_file,mtriple,otriple,
# statement,question_idx,question_type,question,answer,question_es,answer_es,
# generation_errors,corrected_answer,correction_flag,selected_role,
# selection_source,question_translated,validation_status,validation_feedback,
# qa_generation_attempts
QA_CSV = "webnlg_qa_selected_es_v4_validated.csv"

OUT_TRIPLESETS = "ir_triplesets_e5lex.csv"
OUT_TEXTS = "ir_texts_e5lex.csv"
OUT_REL_TRIPLE = "ir_relations_triplesets_e5lex.csv"
OUT_REL_TEXT = "ir_relations_texts_e5lex.csv"

RANDOM_SEED = 42
TRIPLE_SEP = " ||| "

REQUIRE_PARALLEL_REPRESENTATIONS = True

# E5 model.
# Good lighter option: "intfloat/multilingual-e5-base"
# Stronger option: "intfloat/multilingual-e5-large"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_BATCH_SIZE = 64
E5_MAX_SEQ_LENGTH = 512

# E5 normally expects "query:" and "passage:" prefixes.
# For this selection task, we treat tripleset as the "query" and lexicalisation as the "passage".
E5_TRIPLE_PREFIX = "query: "
E5_TEXT_PREFIX = "passage: "


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


def make_tripleset_doc_id(base_id: str, lang: str) -> str:
    return f"tripleset__{base_id}__{lang}"


def make_text_doc_id(base_id: str, lang: str) -> str:
    return f"text__{base_id}__{lang}"


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"  Saved {len(rows):>7} rows → {path}")


def join_triples(triples: List[str]) -> str:
    return TRIPLE_SEP.join([str(t).strip() for t in triples if str(t).strip()])


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
# MERGE / PARALLELIZE ENTRIES
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
        with open("entry_merge_collision_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["base_id", "field", "kept", "seen", "xml_file"],
            )
            writer.writeheader()
            writer.writerows(collision_diagnostics)

        log("  Saved collision diagnostics → entry_merge_collision_diagnostics.csv")

    return out


def filter_to_parallel_entries(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    kept = []
    dropped = []

    for entry in entries:
        reasons = []

        if not entry["mtriples"]:
            reasons.append("missing_en_mtriples")
        if not entry["striples"]:
            reasons.append("missing_es_striples")
        if not entry["lex_en"]:
            reasons.append("missing_en_lex")
        if not entry["lex_es"]:
            reasons.append("missing_es_lex")

        if REQUIRE_PARALLEL_REPRESENTATIONS and reasons:
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
        with open("dropped_non_parallel_entries.csv", "w", newline="", encoding="utf-8") as f:
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

        log(f"  Dropped non-parallel entries: {len(dropped)}")
        log("  Saved dropped entries → dropped_non_parallel_entries.csv")

    log(f"Parallel base entries kept: {len(kept)}")
    return kept, dropped


# ─────────────────────────────────────────────────────────────
# E5 LEXICALISATION SELECTION
# ─────────────────────────────────────────────────────────────

def load_e5_model() -> SentenceTransformer:
    log(f"Loading E5 model: {E5_MODEL_NAME}")

    model = SentenceTransformer(E5_MODEL_NAME)
    model.max_seq_length = E5_MAX_SEQ_LENGTH
    return model


def encode_e5(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def choose_best_lexicalisations_with_e5(entries: List[Dict]) -> None:
    """
    Mutates each entry by adding:
      selected_lex_en_idx
      selected_lex_en
      selected_lex_en_score
      selected_lex_es_idx
      selected_lex_es
      selected_lex_es_score

    EN selection:
      compare mtriples text vs each English lexicalisation.

    ES selection:
      compare striples text vs each Spanish lexicalisation.
    """
    model = load_e5_model()

    pair_rows = []
    triple_texts = []
    lex_texts = []

    for entry_idx, entry in enumerate(entries):
        base_id = make_entry_key_str(entry)

        # English candidates.
        en_triples_text = join_triples(entry["mtriples"])

        for lex_idx, lex in enumerate(entry["lex_en"]):
            pair_rows.append({
                "entry_idx": entry_idx,
                "base_id": base_id,
                "lang": "en",
                "lex_idx": lex_idx,
                "lex_text": lex,
                "triples_text": en_triples_text,
            })
            triple_texts.append(E5_TRIPLE_PREFIX + en_triples_text)
            lex_texts.append(E5_TEXT_PREFIX + lex)

        # Spanish candidates.
        es_triples_text = join_triples(entry["striples"])

        for lex_idx, lex in enumerate(entry["lex_es"]):
            pair_rows.append({
                "entry_idx": entry_idx,
                "base_id": base_id,
                "lang": "es",
                "lex_idx": lex_idx,
                "lex_text": lex,
                "triples_text": es_triples_text,
            })
            triple_texts.append(E5_TRIPLE_PREFIX + es_triples_text)
            lex_texts.append(E5_TEXT_PREFIX + lex)

    log(f"E5 lexicalisation candidate pairs: {len(pair_rows):,}")

    log("Encoding tripleset sides with E5...")
    triple_emb = encode_e5(model, triple_texts, E5_BATCH_SIZE)

    log("Encoding lexicalisation sides with E5...")
    lex_emb = encode_e5(model, lex_texts, E5_BATCH_SIZE)

    # Embeddings are normalized, so dot product = cosine similarity.
    scores = np.sum(triple_emb * lex_emb, axis=1)

    # Select best candidate for each (entry_idx, lang).
    best = {}

    for row, score in zip(pair_rows, scores):
        key = (row["entry_idx"], row["lang"])

        if key not in best or float(score) > best[key]["score"]:
            best[key] = {
                "score": float(score),
                "lex_idx": row["lex_idx"],
                "lex_text": row["lex_text"],
                "triples_text": row["triples_text"],
                "base_id": row["base_id"],
            }

    missing = []

    for entry_idx, entry in enumerate(entries):
        for lang in ("en", "es"):
            key = (entry_idx, lang)

            if key not in best:
                missing.append((make_entry_key_str(entry), lang))
                continue

            if lang == "en":
                entry["selected_lex_en_idx"] = best[key]["lex_idx"]
                entry["selected_lex_en"] = best[key]["lex_text"]
                entry["selected_lex_en_score"] = best[key]["score"]
            else:
                entry["selected_lex_es_idx"] = best[key]["lex_idx"]
                entry["selected_lex_es"] = best[key]["lex_text"]
                entry["selected_lex_es_score"] = best[key]["score"]

    if missing:
        raise ValueError(f"Missing E5 selections for {len(missing)} entry/language pairs. Examples: {missing[:10]}")

    # Save diagnostics.
    diag_path = "e5_lexicalisation_selection_diagnostics.csv"

    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "base_id",
            "lang",
            "lex_idx",
            "score",
            "selected",
            "triples_text",
            "lex_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row, score in zip(pair_rows, scores):
            entry = entries[row["entry_idx"]]
            selected_idx = (
                entry["selected_lex_en_idx"]
                if row["lang"] == "en"
                else entry["selected_lex_es_idx"]
            )

            writer.writerow({
                "base_id": row["base_id"],
                "lang": row["lang"],
                "lex_idx": row["lex_idx"],
                "score": float(score),
                "selected": int(row["lex_idx"] == selected_idx),
                "triples_text": row["triples_text"],
                "lex_text": row["lex_text"],
            })

    log(f"Saved E5 selection diagnostics → {diag_path}")

    en_scores = [e["selected_lex_en_score"] for e in entries]
    es_scores = [e["selected_lex_es_score"] for e in entries]

    print("\nE5 selected lexicalisation similarity:")
    print(f"  EN mean={np.mean(en_scores):.4f}, median={np.median(en_scores):.4f}, min={np.min(en_scores):.4f}, max={np.max(en_scores):.4f}")
    print(f"  ES mean={np.mean(es_scores):.4f}, median={np.median(es_scores):.4f}, min={np.min(es_scores):.4f}, max={np.max(es_scores):.4f}")


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
    "e5_triples_text",
    "e5_similarity",
    "lex_selection_method",
    "e5_model",
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
    log("Loading full WebNLG dataset (all sizes)…")
    raw_entries = load_webnlg_all_sizes(WEBNLG_ROOT)

    log("Merging entries by canonical base key…")
    merged_entries = merge_entries_by_base_key(raw_entries)

    log("Filtering to entries with parallel EN/ES triples and EN/ES texts…")
    all_entries, dropped_entries = filter_to_parallel_entries(merged_entries)

    # ── 3. Select lexicalisations using E5 ────────────────────
    log("Selecting most semantically similar lexicalisation per entry/language using E5…")
    choose_best_lexicalisations_with_e5(all_entries)

    # ── 4. Build documents & relations ───────────────────────
    log("Building parallel documents and relations…")

    triple_docs: List[Dict] = []
    text_docs: List[Dict] = []
    rel_triple: List[Dict] = []
    rel_text: List[Dict] = []

    seen_triple_ids: set = set()
    seen_text_ids: set = set()

    seen_rel_triple: set = set()
    seen_rel_text: set = set()

    for entry in all_entries:
        base_id = make_entry_key_str(entry)
        mtriples = entry["mtriples"]
        striples = entry["striples"]

        matching_qa_rows = get_matching_qa_rows(entry, qa_by_mtriple)

        # Tripleset documents: exactly EN + ES per base entry.
        tripleset_variants = [
            {
                "lang": "en",
                "triple_type": "modifiedtripleset",
                "triples": mtriples,
            },
            {
                "lang": "es",
                "triple_type": "spanishtripleset",
                "triples": striples,
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
                "triples": join_triples(triples),
                "supporting_mtriples": join_triples(mtriples),
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

        # Text documents: exactly EN + ES per base entry, selected by E5.
        text_variants = [
            {
                "lang": "en",
                "lex_idx": entry["selected_lex_en_idx"],
                "text": entry["selected_lex_en"],
                "score": entry["selected_lex_en_score"],
                "triples_text": join_triples(mtriples),
            },
            {
                "lang": "es",
                "lex_idx": entry["selected_lex_es_idx"],
                "text": entry["selected_lex_es"],
                "score": entry["selected_lex_es_score"],
                "triples_text": join_triples(striples),
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
                "supporting_mtriples": join_triples(mtriples),
                "e5_triples_text": variant["triples_text"],
                "e5_similarity": variant["score"],
                "lex_selection_method": "max_e5_tripleset_similarity",
                "e5_model": E5_MODEL_NAME,
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

    # ── 5. Strict checks before writing ───────────────────────
    log("Running strict parallelism checks…")

    triple_base_by_lang = defaultdict(set)
    text_base_by_lang = defaultdict(set)

    for d in triple_docs:
        triple_base_by_lang[d["lang"]].add(d["base_id"])

    for d in text_docs:
        text_base_by_lang[d["lang"]].add(d["base_id"])

    expected_langs = {"en", "es"}

    for lang in expected_langs:
        tset = triple_base_by_lang[lang]
        xset = text_base_by_lang[lang]

        if tset != xset:
            raise ValueError(
                f"Non-parallel base IDs for lang={lang}: "
                f"tripleset_only={len(tset - xset)}, text_only={len(xset - tset)}"
            )

    if triple_base_by_lang["en"] != triple_base_by_lang["es"]:
        raise ValueError("EN and ES tripleset base inventories differ.")

    if text_base_by_lang["en"] != text_base_by_lang["es"]:
        raise ValueError("EN and ES text base inventories differ.")

    triple_counts = Counter((d["lang"], d["size"]) for d in triple_docs)
    text_counts = Counter((d["lang"], d["size"]) for d in text_docs)

    if triple_counts != text_counts:
        raise ValueError(
            "Tripleset/text size distributions differ:\n"
            f"triples={dict(triple_counts)}\n"
            f"texts={dict(text_counts)}"
        )

    log("  Strict parallelism checks passed.")

    # ── 6. Write CSVs ─────────────────────────────────────────
    log("Writing output files…")

    write_csv(OUT_TRIPLESETS, TRIPLE_DOC_FIELDS, triple_docs)
    write_csv(OUT_TEXTS, TEXT_DOC_FIELDS, text_docs)
    write_csv(OUT_REL_TRIPLE, REL_FIELDS, rel_triple)
    write_csv(OUT_REL_TEXT, REL_FIELDS, rel_text)

    # ── 7. Summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("IR DOCUMENT COLLECTION SUMMARY — E5 LEXICALISATION SELECTION")
    print("=" * 80)

    print("\nBase-entry inventory:")
    print(f"  raw XML entries loaded          : {len(raw_entries):,}")
    print(f"  merged canonical base entries   : {len(merged_entries):,}")
    print(f"  parallel base entries kept      : {len(all_entries):,}")
    print(f"  non-parallel base entries drops : {len(dropped_entries):,}")

    print("\nDocument counts:")
    print(f"  tripleset documents : {len(triple_docs):,}")
    print(f"  text documents      : {len(text_docs):,}")

    triple_lang_dist = Counter(d["lang"] for d in triple_docs)
    text_lang_dist = Counter(d["lang"] for d in text_docs)

    print("\nTripleset document language distribution:")
    for lang, count in sorted(triple_lang_dist.items()):
        print(f"  lang={lang} : {count:,}")

    print("\nText document language distribution:")
    for lang, count in sorted(text_lang_dist.items()):
        print(f"  lang={lang} : {count:,}")

    print("\nTripleset document type distribution:")
    triple_type_dist = Counter(d["triple_type"] for d in triple_docs)
    for typ, count in sorted(triple_type_dist.items()):
        print(f"  {typ} : {count:,}")

    print("\nSize distribution by language:")
    all_size_keys = sorted(set(triple_counts) | set(text_counts))
    for lang, size in all_size_keys:
        print(
            f"  lang={lang}, size={size}: "
            f"triplesets={triple_counts[(lang, size)]:,}, "
            f"texts={text_counts[(lang, size)]:,}"
        )

    print("\nE5 selected lexicalisation similarity in final text docs:")
    en_scores = [
        float(d["e5_similarity"])
        for d in text_docs
        if d["lang"] == "en"
    ]
    es_scores = [
        float(d["e5_similarity"])
        for d in text_docs
        if d["lang"] == "es"
    ]

    print(
        f"  EN: mean={np.mean(en_scores):.4f}, median={np.median(en_scores):.4f}, "
        f"min={np.min(en_scores):.4f}, max={np.max(en_scores):.4f}"
    )
    print(
        f"  ES: mean={np.mean(es_scores):.4f}, median={np.median(es_scores):.4f}, "
        f"min={np.min(es_scores):.4f}, max={np.max(es_scores):.4f}"
    )

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

    rel_triple_doc_lang = {d["doc_id"]: d["lang"] for d in triple_docs}
    rel_text_doc_lang = {d["doc_id"]: d["lang"] for d in text_docs}

    rel_triple_lang_dist = Counter(rel_triple_doc_lang.get(r["doc_id"], "unknown") for r in rel_triple)
    rel_text_lang_dist = Counter(rel_text_doc_lang.get(r["doc_id"], "unknown") for r in rel_text)

    print("\nTripleset relation language distribution:")
    for lang, count in sorted(rel_triple_lang_dist.items()):
        print(f"  lang={lang} : {count:,}")

    print("\nText relation language distribution:")
    for lang, count in sorted(rel_text_lang_dist.items()):
        print(f"  lang={lang} : {count:,}")

    covered_triple = len({normalise_triple(r["qa_mtriple"]) for r in rel_triple})
    covered_text = len({normalise_triple(r["qa_mtriple"]) for r in rel_text})

    print(f"\nQA mtriples covered by triplesets : {covered_triple:,}/{len(qa_by_mtriple):,}")
    print(f"QA mtriples covered by texts      : {covered_text:,}/{len(qa_by_mtriple):,}")

    covered_mtriples = {normalise_triple(r["qa_mtriple"]) for r in rel_triple}
    uncovered = [mt for mt in qa_by_mtriple if mt not in covered_mtriples]

    if uncovered:
        print(f"\n⚠ Uncovered QA mtriples ({len(uncovered):,}):")
        for mt in uncovered[:10]:
            print(f"   {mt}")
        if len(uncovered) > 10:
            print(f"   … and {len(uncovered) - 10:,} more")

    print("=" * 80)