#!/usr/bin/env python3
"""Evaluate WebNLG IR runs with binary qrels at canonical tripleset level.

This evaluator uses ALL relevant documents in qrels for each qid. It does not
collapse relevance to one document. Recall/AP/nDCG denominators use the full
number of relevant canonical documents available for that qid.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ir_common import ensure_dir, read_table, table_path, write_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="Run files or glob patterns.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ks", nargs="+", type=int, default=[10, 100])
    return p.parse_args()


def expand_paths(patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        out.extend(matches if matches else [pat])
    out = [p for p in out if Path(p).exists()]
    if not out:
        raise FileNotFoundError("No run files matched --runs")
    return out


def dcg_binary(rels: list[int], k: int) -> float:
    rels = rels[:k]
    return float(sum(rel / np.log2(i + 2) for i, rel in enumerate(rels)))


def idcg_binary(n_rel: int, k: int) -> float:
    return dcg_binary([1] * min(n_rel, k), k)


def per_query_metrics(run: pd.DataFrame, qrels: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    qrels = qrels.copy()
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels = qrels[qrels["relevance"] > 0].drop_duplicates(["qid", "docno"])

    rel_map = qrels.groupby("qid")["docno"].apply(lambda x: set(map(str, x))).to_dict()

    run = run.copy()
    run["qid"] = run["qid"].astype(str)
    run["docno"] = run["docno"].astype(str)

    # Evaluate all qids that this run attempted and for which qrels exist.
    # This prevents accidental test-only evaluation if the run was generated with --split test.
    run_qids = set(run["qid"].unique())
    qids = sorted(set(rel_map.keys()).intersection(run_qids))

    if not qids:
        return pd.DataFrame(columns=["qid", "n_relevant", "AP"] + [f"Recall@{k}" for k in ks])

    run = run[run["qid"].isin(qids)].copy()
    run = run.sort_values(["qid", "rank"])
    # Remove duplicate doc hits for a query, preserving earliest rank.
    run = run.drop_duplicates(["qid", "docno"], keep="first")
    docs_by_qid = run.groupby("qid")["docno"].apply(lambda x: list(map(str, x))).to_dict()

    rows = []
    for qid in qids:
        rels = rel_map[qid]
        ranking = docs_by_qid.get(qid, [])
        n_rel = len(rels)
        row = {"qid": qid, "n_relevant": n_rel}
        binary = [1 if d in rels else 0 for d in ranking]

        hits = 0
        precisions = []
        for i, is_rel in enumerate(binary, start=1):
            if is_rel:
                hits += 1
                precisions.append(hits / i)
        # AP denominator is ALL relevant docs, not only retrieved relevant docs.
        row["AP"] = float(sum(precisions) / n_rel) if n_rel else 0.0

        for k in ks:
            at_k = binary[:k]
            hits_k = sum(at_k)
            row[f"Recall@{k}"] = float(hits_k / n_rel) if n_rel else 0.0
            row[f"P@{k}"] = float(hits_k / k) if k else 0.0
            dcg = dcg_binary(binary, k)
            idcg = idcg_binary(n_rel, k)
            row[f"nDCG@{k}"] = float(dcg / idcg) if idcg > 0 else 0.0
            rr = 0.0
            for i, is_rel in enumerate(at_k, start=1):
                if is_rel:
                    rr = 1.0 / i
                    break
            row[f"MRR@{k}"] = rr
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(perq: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in perq.columns if c in {"AP", "n_relevant"} or "@" in c]
    out = perq.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    counts = perq.groupby(group_cols, dropna=False).size().reset_index(name="n_queries")
    return counts.merge(out, on=group_cols, how="left")


def add_all_groups(perq: pd.DataFrame) -> pd.DataFrame:
    """Create actual + ALL roll-up rows without double-counting queries.

    We create every combination of ALL roll-ups across the reporting dimensions,
    then remove duplicate (run_file, qid, dimensions) rows. This prevents inflated
    n_queries when two different roll-up recipes lead to the same group.
    """
    import itertools

    dimensions = ["question_type3", "query_lang", "doc_lang", "scenario", "split"]
    frames = []
    for r in range(0, len(dimensions) + 1):
        for cols_to_all in itertools.combinations(dimensions, r):
            tmp = perq.copy()
            for col in cols_to_all:
                tmp[col] = "ALL"
            frames.append(tmp)

    expanded = pd.concat(frames, ignore_index=True)
    dedup_key = ["run_file", "qid"] + dimensions
    dedup_key = [c for c in dedup_key if c in expanded.columns]
    expanded = expanded.drop_duplicates(dedup_key, keep="first")
    return expanded


def first_nonnull(df: pd.DataFrame, col: str, default: str) -> str:
    if col in df.columns and df[col].notna().any():
        return str(df[col].dropna().astype(str).iloc[0])
    return default


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    data_dir = Path(args.data_dir)
    qrels = read_table(table_path(data_dir, "qrels"))
    queries = read_table(table_path(data_dir, "queries"))
    qmeta_cols = ["qid", "query_lang", "split", "category", "question_type3", "selected_role"]
    qmeta = queries[qmeta_cols].drop_duplicates("qid")

    perq_all = []
    for path in expand_paths(args.runs):
        run = read_table(path)
        if "qid" not in run.columns or "docno" not in run.columns or "rank" not in run.columns:
            raise ValueError(f"Run {path} must contain qid/docno/rank")
        run["qid"] = run["qid"].astype(str)
        run["docno"] = run["docno"].astype(str)

        method = first_nonnull(run, "method", Path(path).stem)
        representation = first_nonnull(run, "representation", "unknown")
        doc_lang = first_nonnull(run, "doc_lang", "unknown")
        scenario = first_nonnull(run, "scenario", f"unknown_to_{doc_lang}")

        perq = per_query_metrics(run, qrels, sorted(set(args.ks)))
        if perq.empty:
            print(f"WARNING: no evaluable qids in {path}")
            continue
        perq = perq.merge(qmeta, on="qid", how="left")
        perq["method"] = method
        perq["representation"] = representation
        perq["doc_lang"] = doc_lang
        perq["scenario"] = scenario
        perq["run_file"] = path
        perq_all.append(perq)

    if not perq_all:
        raise ValueError("No run produced evaluable qids.")

    perq_all = pd.concat(perq_all, ignore_index=True)
    write_table(perq_all, out_dir / "per_query_metrics.csv")
    try:
        write_table(perq_all, out_dir / "per_query_metrics.parquet")
    except Exception:
        pass

    expanded = add_all_groups(perq_all)
    group_cols = ["method", "representation", "query_lang", "doc_lang", "scenario", "split", "question_type3"]
    summary = summarize(expanded, group_cols)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    try:
        write_table(summary, out_dir / "summary_metrics.parquet")
    except Exception:
        pass

    sort_cols = ["method", "representation", "query_lang", "doc_lang", "scenario", "split", "question_type3"]
    print(summary.sort_values(sort_cols).to_string(index=False))
    print(f"Saved evaluation to {out_dir}")


if __name__ == "__main__":
    main()
