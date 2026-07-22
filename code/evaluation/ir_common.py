from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Query languages are real natural-language query fields.
QUERY_LANGS = {"en", "ca", "es"}

# Document-language labels include the experimental mixed-language tripleset
# condition. In this package, doc_lang="mix" means:
#   triples_mix = mixed EN/ES triples
#   text_mix    = English lexicalisation duplicated by the lang3 mapper
#   concat_mix  = triples_mix + text_mix
DOC_LANGS = {"en", "es", "mix"}

# Backward-compatible name used by older scripts.
LANGS = DOC_LANGS

ALL_SPLIT_VALUES = {"ALL", "*", "NONE", "NULL", ""}


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df.to_csv(path, index=False, sep=sep)
    else:
        df.to_parquet(path, index=False)


def table_path(data_dir: str | Path, stem: str) -> Path:
    data_dir = Path(data_dir)
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {data_dir}")


def is_all_split(split: str | None) -> bool:
    if split is None:
        return True
    return str(split).upper() in ALL_SPLIT_VALUES


def representation_column(representation: str, doc_lang: Optional[str] = None) -> str:
    """
    Map a representation and document-language condition to a prepared document
    text column.

    Supported document-language labels:
      en  -> English/canonical triples + English text
      es  -> Spanish triples + Spanish text
      mix -> mixed EN/ES triples + English text
    """
    if doc_lang not in DOC_LANGS:
        raise ValueError("All representations require --doc_lang in {'en','es','mix'}")

    if representation == "triples":
        return f"triples_{doc_lang}"
    if representation == "text":
        return f"text_{doc_lang}"
    if representation == "concat":
        return f"concat_{doc_lang}"
    raise ValueError(f"Unknown representation: {representation}")


def effective_doc_lang(representation: str, doc_lang: Optional[str]) -> str:
    if doc_lang not in DOC_LANGS:
        raise ValueError(f"representation={representation} requires --doc_lang en, es, or mix")
    return str(doc_lang)


def load_docs_for_run(data_dir: str | Path, representation: str, doc_lang: Optional[str] = None) -> pd.DataFrame:
    docs = read_table(table_path(data_dir, "docs"))
    col = representation_column(representation, doc_lang)
    if col not in docs.columns:
        raise ValueError(f"Missing document field {col} in docs table")

    keep_cols = ["docno", "split", "category", "eid", "size", "n_triples", col]
    keep_cols = [c for c in keep_cols if c in docs.columns]
    out = docs[keep_cols].copy()
    out = out.rename(columns={col: "text"})
    out["text"] = out["text"].fillna("").astype(str)
    out = out[out["text"].str.len() > 0].copy()
    out["doc_lang"] = effective_doc_lang(representation, doc_lang)
    return out


def load_queries_for_run(data_dir: str | Path, query_lang: str, split: str = "ALL") -> pd.DataFrame:
    if query_lang not in QUERY_LANGS:
        raise ValueError("query_lang must be 'en' or 'es'")
    q = read_table(table_path(data_dir, "queries"))
    q = q[q["query_lang"] == query_lang].copy()
    if not is_all_split(split):
        q = q[q["split"] == split].copy()
    q["query"] = q["query"].fillna("").astype(str)
    return q


def save_run(
    df: pd.DataFrame,
    output_path: str | Path,
    method: str,
    representation: str,
    query_lang: str,
    doc_lang: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    df = df.copy()
    if query_lang not in QUERY_LANGS:
        raise ValueError("query_lang must be 'en' or 'es'")
    if doc_lang not in DOC_LANGS:
        raise ValueError("doc_lang must be 'en', 'es', or 'mix'")
    df["method"] = method
    df["representation"] = representation
    df["query_lang"] = query_lang
    df["doc_lang"] = str(doc_lang)
    df["scenario"] = df["query_lang"].astype(str) + "_to_" + df["doc_lang"].astype(str)
    if extra:
        for k, v in extra.items():
            df[k] = v

    cols = [
        "qid", "docno", "score", "rank",
        "method", "representation", "query_lang", "doc_lang", "scenario",
        "query_id_base", "split", "category", "question_type3", "selected_role",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    extra_cols = [c for c in df.columns if c not in cols]
    write_table(df[cols + extra_cols], output_path)
    print(f"Saved {len(df):,} rows to {output_path}")


def attach_query_metadata(run: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "qid", "query_id_base", "split", "category", "question_type3", "selected_role",
        "query_lang", "mtriple",
    ]
    meta_cols = [c for c in meta_cols if c in queries.columns]
    return run.merge(
        queries[meta_cols].drop_duplicates("qid"),
        on="qid",
        how="left",
        suffixes=("", "_q"),
    )


def topk_from_scores(qids: np.ndarray, docnos: np.ndarray, scores: np.ndarray, top_k: int) -> pd.DataFrame:
    """scores shape: n_queries x n_docs"""
    n_q, n_d = scores.shape
    k = min(top_k, n_d)
    if k <= 0:
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_idx = np.arange(n_q)[:, None]
    top_scores = scores[row_idx, idx]
    order = np.argsort(-top_scores, axis=1)
    final_idx = idx[row_idx, order]
    final_scores = top_scores[row_idx, order]
    return pd.DataFrame({
        "qid": np.repeat(qids, k),
        "docno": docnos[final_idx.reshape(-1)],
        "score": final_scores.reshape(-1),
        "rank": np.tile(np.arange(1, k + 1), n_q),
    })


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\wáéíóúüñ]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]


def iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield start, df.iloc[start:start + batch_size]
