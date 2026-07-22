#!/usr/bin/env python3
"""
Prepare WebNLG_ES IR data for triples-vs-text-vs-hybrid evaluation.

Lang3-compatible behavior
-------------------------
The prepared document table has one row per canonical WebNLG base entry
(`docno = base_id`) and can contain three document-language conditions:

  en   = English/canonical triples + English lexicalisation
  es   = Spanish triples + Spanish lexicalisation
  mix  = mixed EN/ES triples + English lexicalisation

The resulting fields are:

  triples_en, triples_es, triples_mix
  text_en,    text_es,    text_mix
  concat_en,  concat_es,  concat_mix

Queries remain bilingual only:

  query_lang = en from QA.question
  query_lang = es from QA.question_es

`--min_triples` filters QUESTION ELIGIBILITY by requiring at least one relevant
base document with size >= min_triples. Qrels then include ALL relevant base
documents in the retrieval pool for every retained query.

Call:
python prepare_webnlg_ir.py \
  --qa_csv './webnlg_qa_selected_es_ca_v4_validated.csv' \
  --triplesets_csv './dataset_lang3/ir_triplesets_lang3_min1.csv' \
  --texts_csv './dataset_lang3/ir_texts_lang3_min1.csv' \
  --out_dir ./prepared_min4_mixed_normtext \
  --min_triples 4 \
  --triple_variant normalized \
  --text_variant normalized
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from ir_common import ensure_dir, write_table


DOC_LANGS = ["en", "es", "mix"]
QUERY_LANGS = ["en", "es", "ca"]
TRIPLE_SEP = " ||| "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qa_csv", required=True)
    p.add_argument("--triplesets_csv", required=True)
    p.add_argument("--texts_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--min_triples",
        type=int,
        default=1,
        help=(
            "Minimum relevant tripleset size used to keep a QA item. "
            "This does not filter the document pool unless --filter_docs_by_min_triples is passed."
        ),
    )
    p.add_argument(
        "--filter_docs_by_min_triples",
        action="store_true",
        help="Optional ablation: also remove documents with size < --min_triples from the retrieval pool.",
    )
    p.add_argument(
        "--valid_statuses",
        nargs="+",
        default=["valid", "validator_fixed"],
        help="QA validation_status values to keep. Use ALL to keep everything.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["ALL"],
        help="QA splits to keep. Default ALL keeps train+dev+test.",
    )
    p.add_argument(
        "--doc_langs",
        nargs="+",
        choices=DOC_LANGS,
        default=DOC_LANGS,
        help="Document-language fields expected in triples/text CSVs and prepared docs.",
    )
    p.add_argument(
        "--triple_variant",
        choices=["raw", "normalized", "both"],
        default="both",
        help="How to create triples_* retrieval fields.",
    )
    p.add_argument(
        "--text_variant",
        choices=["raw", "normalized", "both"],
        default="raw",
        help=(
            "How to create text_* retrieval fields. "
            "raw keeps original lexicalisations; normalized applies light retrieval-oriented text normalization; "
            "both indexes raw text followed by its normalized version."
        ),
    )
    p.add_argument(
        "--normalize_texts",
        action="store_true",
        help="Shortcut for --text_variant normalized.",
    )
    p.add_argument(
        "--file_format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output table format. Use csv if pyarrow/fastparquet is unavailable.",
    )
    p.add_argument(
        "--require_parallel_docs",
        action="store_true",
        default=True,
        help="Keep only base entries with triples/text fields for all requested --doc_langs. Enabled by default.",
    )
    p.add_argument("--allow_nonparallel_docs", dest="require_parallel_docs", action="store_false")
    p.add_argument("--drop_queries_without_qrels", action="store_true", default=True)
    p.add_argument("--keep_queries_without_qrels", dest="drop_queries_without_qrels", action="store_false")
    return p.parse_args()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def split_camel(token: str) -> str:
    token = re.sub(r"([a-záéíóúüñ])([A-ZÁÉÍÓÚÜÑ])", r"\1 \2", token)
    token = re.sub(r"([A-ZÁÉÍÓÚÜÑ]+)([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ])", r"\1 \2", token)
    return token


def normalize_triple_text(s: str) -> str:
    """Normalize WebNLG triples for textual retrieval while preserving content."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("|||", " . ")
    s = s.replace("||", " . ")
    s = s.replace("|", " ")
    s = s.replace("_", " ")
    s = re.sub(r"@\w+", " ", s)
    s = s.replace('"', " ")
    s = split_camel(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\s\.,;:()\-/áéíóúüñÁÉÍÓÚÜÑ]", " ", s)
    return normalize_ws(s)


def normalize_text_text(s: str) -> str:
    """
    Lightly normalize lexicalised text for retrieval ablations.

    No stopword removal, stemming, lemmatization, or translation is applied.
    """
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("_", " ")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.lower()
    s = re.sub(r"[^\w\s\.,;:()\-/áéíóúüñ]", " ", s, flags=re.UNICODE)
    return normalize_ws(s)


def qtype3(row: pd.Series) -> str:
    role = str(row.get("selected_role", ""))
    if role == "yes_q":
        return "yes"
    if role == "no_q":
        return "no"
    if role.startswith("extractive"):
        return "extractive"
    qt = str(row.get("question_type", ""))
    return "extractive" if qt == "extractive" else qt


def infer_base_id_from_doc_id(doc_id: str, kind: str, doc_langs: list[str]) -> str:
    """
    Accepts IDs like:
      tripleset__train__Airport__Id1__2triples__mix
      text__train__Airport__Id1__2triples__en
    and returns:
      train__Airport__Id1__2triples
    """
    x = str(doc_id)
    if kind == "tripleset":
        x = x.replace("tripleset__", "", 1)
        parts = x.split("__")
        if parts[-1] in doc_langs:
            parts = parts[:-1]
        return "__".join(parts[:4])
    if kind == "text":
        x = x.replace("text__", "", 1)
        parts = x.split("__")
        if parts[-1] in doc_langs:
            parts = parts[:-1]
        return "__".join(parts[:4])
    raise ValueError(kind)


def ensure_base_id(df: pd.DataFrame, kind: str, doc_langs: list[str]) -> pd.DataFrame:
    df = df.copy()
    if "base_id" not in df.columns:
        df["base_id"] = df["doc_id"].map(lambda x: infer_base_id_from_doc_id(x, kind, doc_langs))
    df["base_id"] = df["base_id"].astype(str)
    return df


def infer_lang(df: pd.DataFrame, kind: str, doc_langs: list[str]) -> pd.DataFrame:
    df = df.copy()
    if "lang" in df.columns:
        df["lang"] = df["lang"].astype(str).str.lower()
        return df
    if "doc_lang" in df.columns:
        df = df.rename(columns={"doc_lang": "lang"})
        df["lang"] = df["lang"].astype(str).str.lower()
        return df
    lang_re = "|".join(map(re.escape, doc_langs))
    df["lang"] = df["doc_id"].astype(str).str.extract(rf"__({lang_re})(?:__\d+)?$")[0]
    if df["lang"].isna().any():
        raise ValueError(f"Could not infer lang for some {kind} rows. Add a lang column.")
    return df


def infer_n_triples(df: pd.DataFrame) -> pd.Series:
    if "n_triples" in df.columns:
        return pd.to_numeric(df["n_triples"], errors="coerce").fillna(df.get("size", 1)).astype(int)
    if "size" in df.columns:
        return pd.to_numeric(df["size"], errors="coerce").fillna(1).astype(int)
    raise ValueError("Triplesets CSV must contain either 'size' or 'n_triples'.")


def join_unique(xs: Iterable[str]) -> str:
    vals = []
    seen = set()
    for x in xs:
        x = normalize_ws(x)
        if x and x not in seen:
            seen.add(x)
            vals.append(x)
    return "\n".join(vals)


def make_triples_field(raw: str, variant: str) -> str:
    norm = normalize_triple_text(raw)
    raw = normalize_ws(raw)
    if variant == "raw":
        return raw
    if variant == "normalized":
        return norm
    return normalize_ws(raw + "\n" + norm)


def make_text_field(raw: str, variant: str) -> str:
    norm = normalize_text_text(raw)
    raw = normalize_ws(raw)
    if variant == "raw":
        return raw
    if variant == "normalized":
        return norm
    return normalize_ws(raw + "\n" + norm)


def rename_lang_columns(df: pd.DataFrame, prefix: str, suffix: str = "") -> pd.DataFrame:
    mapping = {lang: f"{prefix}_{lang}{suffix}" for lang in DOC_LANGS}
    return df.rename(columns=mapping)


def build_docs(
    triplesets_csv: str,
    texts_csv: str,
    min_triples: int,
    triple_variant: str,
    text_variant: str,
    filter_docs_by_min_triples: bool,
    require_parallel_docs: bool,
    doc_langs: list[str],
) -> tuple[pd.DataFrame, dict]:
    triples = pd.read_csv(triplesets_csv).copy()
    texts = pd.read_csv(texts_csv).copy()

    required_triples = {"doc_id", "split", "category", "eid", "size", "triples"}
    missing_triples = sorted(required_triples - set(triples.columns))
    if missing_triples:
        raise ValueError(f"Triplesets CSV missing columns: {missing_triples}")

    required_texts = {"doc_id", "split", "category", "eid", "size", "text"}
    missing_texts = sorted(required_texts - set(texts.columns))
    if missing_texts:
        raise ValueError(f"Texts CSV missing columns: {missing_texts}")

    triples = ensure_base_id(infer_lang(triples, "tripleset", doc_langs), "tripleset", doc_langs)
    texts = ensure_base_id(infer_lang(texts, "text", doc_langs), "text", doc_langs)
    triples = triples[triples["lang"].isin(doc_langs)].copy()
    texts = texts[texts["lang"].isin(doc_langs)].copy()
    triples["n_triples"] = infer_n_triples(triples)
    triples["size"] = pd.to_numeric(triples["size"], errors="coerce").fillna(triples["n_triples"]).astype(int)
    texts["size"] = pd.to_numeric(texts["size"], errors="coerce").fillna(1).astype(int)

    if "supporting_mtriples" not in triples.columns:
        triples["supporting_mtriples"] = triples["triples"]

    meta_cols = ["base_id", "split", "category", "eid", "size", "n_triples"]
    if "xml_file" in triples.columns:
        meta_cols.append("xml_file")
    meta = triples.sort_values(["base_id", "lang"]).drop_duplicates("base_id")[meta_cols].copy()

    tri_agg = (
        triples.groupby(["base_id", "lang"], as_index=False)
        .agg(
            triples_raw=("triples", join_unique),
            supporting_mtriples=("supporting_mtriples", join_unique),
            n_triples=("n_triples", "max"),
        )
    )
    tri_wide = tri_agg.pivot(index="base_id", columns="lang", values="triples_raw").reset_index()
    tri_wide.columns.name = None
    tri_wide = rename_lang_columns(tri_wide, "triples", "_raw")

    sup_wide = tri_agg.pivot(index="base_id", columns="lang", values="supporting_mtriples").reset_index()
    sup_wide.columns.name = None
    sup_wide = rename_lang_columns(sup_wide, "supporting_mtriples")

    txt_agg = (
        texts.groupby(["base_id", "lang"], as_index=False)
        .agg(text=("text", join_unique), n_lexicalisations=("text", "nunique"))
    )
    txt_wide = txt_agg.pivot(index="base_id", columns="lang", values="text").reset_index()
    txt_wide.columns.name = None
    txt_wide = rename_lang_columns(txt_wide, "text")

    cnt_wide = txt_agg.pivot(index="base_id", columns="lang", values="n_lexicalisations").reset_index()
    cnt_wide.columns.name = None
    cnt_wide = rename_lang_columns(cnt_wide, "n_lex")

    docs = meta.merge(tri_wide, on="base_id", how="left")
    docs = docs.merge(sup_wide, on="base_id", how="left")
    docs = docs.merge(txt_wide, on="base_id", how="left")
    docs = docs.merge(cnt_wide, on="base_id", how="left")

    for lang in doc_langs:
        for c in [f"triples_{lang}_raw", f"supporting_mtriples_{lang}", f"text_{lang}"]:
            if c not in docs.columns:
                docs[c] = ""
            docs[c] = docs[c].fillna("").astype(str)
        c = f"n_lex_{lang}"
        if c not in docs.columns:
            docs[c] = 0
        docs[c] = pd.to_numeric(docs[c], errors="coerce").fillna(0).astype(int)

    before_parallel = len(docs)
    parallel_mask = pd.Series(True, index=docs.index)
    for lang in doc_langs:
        parallel_mask &= docs[f"triples_{lang}_raw"].str.strip().ne("")
        parallel_mask &= docs[f"text_{lang}"].str.strip().ne("")

    if require_parallel_docs:
        dropped = docs[~parallel_mask].copy()
        docs = docs[parallel_mask].copy()
    else:
        dropped = docs.iloc[0:0].copy()

    if filter_docs_by_min_triples:
        docs = docs[docs["n_triples"] >= min_triples].copy()

    document_columns = []
    for lang in doc_langs:
        docs[f"triples_{lang}"] = docs[f"triples_{lang}_raw"].map(lambda x: make_triples_field(x, triple_variant))
        docs[f"text_{lang}_raw"] = docs[f"text_{lang}"].fillna("").astype(str).map(normalize_ws)
        docs[f"text_{lang}_normalized"] = docs[f"text_{lang}_raw"].map(normalize_text_text)
        docs[f"text_{lang}"] = docs[f"text_{lang}_raw"].map(lambda x: make_text_field(x, text_variant))
        docs[f"concat_{lang}"] = (
            "[TRIPLES]\n" + docs[f"triples_{lang}"].fillna("") + "\n[TEXT]\n" + docs[f"text_{lang}"].fillna("")
        ).map(normalize_ws)
        document_columns.extend([
            f"triples_{lang}", f"text_{lang}", f"concat_{lang}",
            f"text_{lang}_raw", f"text_{lang}_normalized",
        ])

    docs = docs.rename(columns={"base_id": "docno"})

    stats = {
        "n_base_docs_before_parallel_filter": int(before_parallel),
        "n_base_docs_after_parallel_filter": int(len(docs) if require_parallel_docs else before_parallel),
        "n_base_docs_dropped_nonparallel": int(len(dropped)),
        "require_parallel_docs": bool(require_parallel_docs),
        "doc_langs": doc_langs,
        "text_variant": text_variant,
        "document_columns": document_columns,
    }
    return docs, stats


def split_triple_field(s: str) -> list[str]:
    s = str(s or "")
    if TRIPLE_SEP in s:
        parts = s.split(TRIPLE_SEP)
    elif "|||" in s:
        parts = s.split("|||")
    else:
        parts = s.split("\n")
    return [normalize_ws(p) for p in parts if normalize_ws(p)]


def build_triple_to_docs(docs: pd.DataFrame) -> dict[str, list[str]]:
    """Map canonical mtriple -> all relevant base docnos."""
    triple_to_docs: dict[str, set[str]] = {}
    # Relevance remains grounded in canonical EN modified triples.
    fallback_support_cols = [c for c in docs.columns if c.startswith("supporting_mtriples_")]
    support_col = "supporting_mtriples_en" if "supporting_mtriples_en" in docs.columns else fallback_support_cols[0]
    raw_col = "triples_en_raw" if "triples_en_raw" in docs.columns else None
    for _, d in tqdm(docs.iterrows(), total=len(docs), desc="Indexing mtriple→all relevant docs"):
        supporting = d.get(support_col, "")
        if not supporting and raw_col:
            supporting = d.get(raw_col, "")
        for part in split_triple_field(supporting):
            triple_to_docs.setdefault(part, set()).add(str(d["docno"]))
    return {k: sorted(v) for k, v in triple_to_docs.items()}


def apply_qa_filters(qa: pd.DataFrame, valid_statuses: List[str], splits: List[str]) -> pd.DataFrame:
    qa = qa.copy()
    if not (len(valid_statuses) == 1 and valid_statuses[0].upper() == "ALL"):
        qa = qa[qa["validation_status"].isin(valid_statuses)].copy()
    if not (len(splits) == 1 and splits[0].upper() == "ALL"):
        qa = qa[qa["split"].isin(splits)].copy()
    return qa


def build_query_id_base(row: pd.Series) -> str:
    lex_id = str(row.get("lex_id", ""))
    qidx = str(row.get("question_idx", ""))
    return (
        f"qa__{int(row.qa_row_id)}__{row.split}__{row.category}__{row.eid}__"
        f"lex{lex_id}__q{qidx}__{row.question_type3}"
    )


def build_queries_and_qrels(
    qa_csv: str,
    docs: pd.DataFrame,
    min_triples: int,
    valid_statuses: List[str],
    splits: List[str],
    drop_queries_without_qrels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    qa = pd.read_csv(qa_csv).reset_index(drop=False).rename(columns={"index": "qa_row_id"})

    # Query-language mapping.
    # English uses QA.question, Spanish uses QA.question_es, Catalan uses QA.question_ca.
    query_columns = {
        "en": "question",
        "es": "question_es",
        "ca": "question_ca",
    }

    required_qa = {
        "split", "category", "eid", "lex_id", "question_idx", "question_type",
        "mtriple", *query_columns.values(),
    }

    missing_qa = sorted(required_qa - set(qa.columns))
    if missing_qa:
        raise ValueError(f"QA CSV missing columns: {missing_qa}")

    qa = apply_qa_filters(qa, valid_statuses, splits)
    qa["question_type3"] = qa.apply(qtype3, axis=1)
    qa["query_id_base"] = qa.apply(build_query_id_base, axis=1)

    triple_to_docs = build_triple_to_docs(docs)
    doc_meta = docs.set_index("docno")[["n_triples"]].to_dict("index")

    qrel_rows = []
    eligible_bases = set()
    excluded_no_rel = 0
    excluded_min = 0

    for _, r in tqdm(
        qa.iterrows(),
        total=len(qa),
        desc="Creating all-doc qrels + filtering eligible questions",
    ):
        mtriple = normalize_ws(r["mtriple"])
        rel_docs_all = triple_to_docs.get(mtriple, [])

        if not rel_docs_all:
            excluded_no_rel += 1
            continue

        rel_docs_ge_min = [
            d for d in rel_docs_all
            if int(doc_meta[d]["n_triples"]) >= min_triples
        ]

        if not rel_docs_ge_min:
            excluded_min += 1
            continue

        eligible_bases.add(r["query_id_base"])

        # Create qrels only for query languages whose question text exists.
        for qlang in QUERY_LANGS:
            qcol = query_columns[qlang]
            qtext = normalize_ws(r.get(qcol, ""))

            if not qtext:
                continue

            qid = f"{r['query_id_base']}__{qlang}"

            for docno in rel_docs_all:
                qrel_rows.append({
                    "qid": qid,
                    "docno": docno,
                    "relevance": 1,
                })

    qrels = pd.DataFrame(qrel_rows, columns=["qid", "docno", "relevance"])
    qids_with_qrels = set(qrels["qid"].unique()) if len(qrels) else set()

    query_rows = []
    qa_eligible = qa[qa["query_id_base"].isin(eligible_bases)].copy()

    for _, r in qa_eligible.iterrows():
        for lang in QUERY_LANGS:
            col = query_columns[lang]
            query_text = normalize_ws(r.get(col, ""))

            if not query_text:
                continue

            qid = f"{r['query_id_base']}__{lang}"

            if drop_queries_without_qrels and qid not in qids_with_qrels:
                continue

            query_rows.append({
                "qid": qid,
                "query_id_base": r["query_id_base"],
                "qa_row_id": int(r["qa_row_id"]),
                "query_lang": lang,
                "query": query_text,
                "split": r["split"],
                "category": r["category"],
                "eid": r["eid"],
                "lex_id": r.get("lex_id", ""),
                "question_idx": r.get("question_idx", ""),
                "question_type": r["question_type"],
                "question_type3": r["question_type3"],
                "selected_role": r.get("selected_role", ""),
                "mtriple": r["mtriple"],
                "answer": r.get("answer", ""),
                "answer_es": r.get("answer_es", ""),
                "answer_ca": r.get("answer_ca", ""),
                "validation_status": r.get("validation_status", ""),
            })

    queries = pd.DataFrame(query_rows)

    if len(qrels):
        qrels = (
            qrels[qrels["qid"].isin(set(queries["qid"]))]
            .drop_duplicates()
            .copy()
        )

    stats = {
        "n_qa_after_status_split_filters": int(len(qa)),
        "n_qa_items_eligible_after_min_triples_filter": int(len(qa_eligible)),
        "n_qa_items_excluded_no_relevant_docs": int(excluded_no_rel),
        "n_qa_items_excluded_by_min_triples_filter": int(excluded_min),
        "query_columns": query_columns,
    }

    return queries, qrels, stats


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if args.normalize_texts:
        args.text_variant = "normalized"

    docs, doc_stats = build_docs(
        args.triplesets_csv,
        args.texts_csv,
        args.min_triples,
        args.triple_variant,
        args.text_variant,
        args.filter_docs_by_min_triples,
        args.require_parallel_docs,
        args.doc_langs,
    )
    queries, qrels, q_stats = build_queries_and_qrels(
        args.qa_csv,
        docs,
        args.min_triples,
        args.valid_statuses,
        args.splits,
        args.drop_queries_without_qrels,
    )

    suffix = ".parquet" if args.file_format == "parquet" else ".csv"
    write_table(docs, out_dir / f"docs{suffix}")
    write_table(queries, out_dir / f"queries{suffix}")
    write_table(qrels, out_dir / f"qrels{suffix}")

    doc_columns = doc_stats.pop("document_columns", [])
    manifest = {
        "min_triples_question_filter": args.min_triples,
        "filter_docs_by_min_triples": bool(args.filter_docs_by_min_triples),
        "valid_statuses": args.valid_statuses,
        "splits": args.splits,
        "doc_langs": args.doc_langs,
        "query_langs": QUERY_LANGS,
        "triple_variant": args.triple_variant,
        "text_variant": args.text_variant,
        "n_docs": int(len(docs)),
        "n_queries": int(len(queries)),
        "n_qrels": int(len(qrels)),
        "n_qids_with_qrels": int(qrels["qid"].nunique()) if len(qrels) else 0,
        "docs_by_size": {str(k): int(v) for k, v in docs["n_triples"].value_counts().sort_index().items()},
        "queries_by_lang": {str(k): int(v) for k, v in queries["query_lang"].value_counts().sort_index().items()} if len(queries) else {},
        "queries_by_type": {str(k): int(v) for k, v in queries["question_type3"].value_counts().sort_index().items()} if len(queries) else {},
        "qrels_relevance_note": "qrels include all relevant canonical base documents in the retrieval pool for retained queries",
        "document_columns": doc_columns,
        **doc_stats,
        **q_stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
