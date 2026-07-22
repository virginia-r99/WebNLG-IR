#!/usr/bin/env python3
"""Aggregate per-run .profile.json files into resource summary CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True, help="Directory containing *.profile.json files.")
    p.add_argument("--out_csv", required=True, help="Path for the detailed resource profile table.")
    p.add_argument("--pattern", default="*.profile.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.runs_dir).glob(args.pattern))
    if not paths:
        raise ValueError(f"No profile files found in {args.runs_dir} with pattern {args.pattern!r}")

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["profile_path"] = str(path)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derived columns. Only create them when the necessary source columns exist.
    if {"doc_encode_time_s", "n_docs"}.issubset(df.columns):
        df["doc_encode_ms_per_doc"] = df["doc_encode_time_s"] / df["n_docs"].replace(0, pd.NA) * 1000
    if {"index_time_s", "n_docs"}.issubset(df.columns):
        df["index_ms_per_doc"] = df["index_time_s"] / df["n_docs"].replace(0, pd.NA) * 1000
    if {"query_encode_time_s", "n_queries"}.issubset(df.columns):
        df["query_encode_ms_per_query"] = df["query_encode_time_s"] / df["n_queries"].replace(0, pd.NA) * 1000
    if {"retrieval_time_s", "n_queries"}.issubset(df.columns):
        df["retrieval_ms_per_query"] = df["retrieval_time_s"] / df["n_queries"].replace(0, pd.NA) * 1000
    if {"scoring_time_s", "n_queries"}.issubset(df.columns):
        df["scoring_ms_per_query"] = df["scoring_time_s"] / df["n_queries"].replace(0, pd.NA) * 1000
    if {"rerank_time_s", "n_candidate_pairs"}.issubset(df.columns):
        df["rerank_ms_per_candidate_pair"] = df["rerank_time_s"] / df["n_candidate_pairs"].replace(0, pd.NA) * 1000
    if {"fusion_time_s", "n_queries"}.issubset(df.columns):
        df["fusion_ms_per_query"] = df["fusion_time_s"] / df["n_queries"].replace(0, pd.NA) * 1000

    front_cols = [
        "method", "representation", "query_lang", "doc_lang", "split", "top_k",
        "n_docs", "n_queries", "n_results", "n_candidate_pairs",
        "total_time_s", "load_data_time_s", "load_runs_time_s", "load_candidates_time_s",
        "model_load_time_s", "index_time_s", "doc_encode_time_s", "query_encode_time_s",
        "query_encode_and_scoring_time_s", "query_encode_and_rerank_time_s",
        "retrieval_time_s", "scoring_time_s", "rerank_time_s", "fusion_time_s",
        "metadata_attach_time_s", "write_time_s",
        "index_ms_per_doc", "doc_encode_ms_per_doc", "query_encode_ms_per_query",
        "retrieval_ms_per_query", "scoring_ms_per_query", "rerank_ms_per_candidate_pair",
        "fusion_ms_per_query",
        "docs_per_second_total", "queries_per_second_total",
        "peak_cpu_rss_mb", "peak_gpu_allocated_mb", "peak_gpu_reserved_mb", "gpu_name", "device",
    ]
    existing_front = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_front]
    df = df[existing_front + other_cols]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed resource summary: {out_csv} ({len(df):,} rows)")

    group_cols = [c for c in ["method", "representation"] if c in df.columns]
    if group_cols:
        agg_spec = {
            "runs": ("profile_path", "count"),
            "mean_total_time_s": ("total_time_s", "mean"),
            "median_total_time_s": ("total_time_s", "median"),
            "mean_peak_cpu_rss_mb": ("peak_cpu_rss_mb", "mean"),
        }
        if "peak_gpu_allocated_mb" in df.columns:
            agg_spec["mean_peak_gpu_allocated_mb"] = ("peak_gpu_allocated_mb", "mean")
        if "doc_encode_ms_per_doc" in df.columns:
            agg_spec["mean_doc_encode_ms_per_doc"] = ("doc_encode_ms_per_doc", "mean")
        if "index_ms_per_doc" in df.columns:
            agg_spec["mean_index_ms_per_doc"] = ("index_ms_per_doc", "mean")
        if "query_encode_ms_per_query" in df.columns:
            agg_spec["mean_query_encode_ms_per_query"] = ("query_encode_ms_per_query", "mean")
        if "scoring_ms_per_query" in df.columns:
            agg_spec["mean_scoring_ms_per_query"] = ("scoring_ms_per_query", "mean")
        if "retrieval_ms_per_query" in df.columns:
            agg_spec["mean_retrieval_ms_per_query"] = ("retrieval_ms_per_query", "mean")
        if "rerank_ms_per_candidate_pair" in df.columns:
            agg_spec["mean_rerank_ms_per_candidate_pair"] = ("rerank_ms_per_candidate_pair", "mean")
        if "fusion_ms_per_query" in df.columns:
            agg_spec["mean_fusion_ms_per_query"] = ("fusion_ms_per_query", "mean")

        compact = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
        compact_path = out_csv.with_name(out_csv.stem + "_by_method_representation.csv")
        compact.to_csv(compact_path, index=False)
        print(f"Saved compact resource summary: {compact_path} ({len(compact):,} rows)")


if __name__ == "__main__":
    main()
