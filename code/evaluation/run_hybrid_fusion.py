#!/usr/bin/env python3
"""Late representation fusion for triples + one monolingual text run.

Supports:
  - weighted RRF: robust rank fusion, primary hybrid.
  - weighted score fusion: calibrated score fusion after per-query normalization.
  - union candidates: candidate pool for multi-vector cascade reranking.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import numpy as np
import pandas as pd

from ir_common import read_table, save_run
from resource_profiler import ResourceProfiler, profile_path_from_output, safe_ms_per_item, safe_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_triples", required=True)
    p.add_argument("--run_text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fusion", choices=["rrf", "score", "union"], default="rrf")
    p.add_argument("--weight_triples", type=float, default=0.5)
    p.add_argument("--weight_text", type=float, default=0.5)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--candidate_top_n", type=int, default=1000, help="For --fusion union, keep this many from each source before union.")
    p.add_argument("--method", default=None, help="Override method name in output. By default inferred from input run.")
    p.add_argument("--query_lang", choices=["en", "ca", "es"], default=None)
    p.add_argument("--doc_lang", choices=["en", "es", "mix"], default=None, help="Output document language/condition of the hybrid scenario.")
    p.add_argument("--profile", action="store_true", help="Write a .profile.json resource log next to the run file.")
    p.add_argument("--profile_path", default=None, help="Optional explicit path for the resource profile JSON.")
    return p.parse_args()


def load_run(path: str, source: str, top_n: int | None = None) -> pd.DataFrame:
    df = read_table(path).copy()
    needed = {"qid", "docno", "score", "rank"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    if top_n is not None:
        df = df.sort_values(["qid", "rank"]).groupby("qid", as_index=False).head(top_n)
    df["source"] = source
    return df


def infer_meta(tri: pd.DataFrame, txt: pd.DataFrame, method_override: str | None, query_lang_override: str | None, doc_lang_override: str | None) -> tuple[str, str, str]:
    if method_override:
        method = method_override
    else:
        methods = sorted(set(tri.get("method", pd.Series(dtype=str)).dropna().astype(str)).union(
                         set(txt.get("method", pd.Series(dtype=str)).dropna().astype(str))))
        method = methods[0] if len(methods) == 1 else "hybrid"

    if query_lang_override:
        qlang = query_lang_override
    else:
        langs = sorted(set(tri.get("query_lang", pd.Series(dtype=str)).dropna().astype(str)).union(
                       set(txt.get("query_lang", pd.Series(dtype=str)).dropna().astype(str))))
        qlang = langs[0] if len(langs) == 1 else "mixed"

    if doc_lang_override:
        dlang = doc_lang_override
    else:
        txt_langs = sorted(set(txt.get("doc_lang", pd.Series(dtype=str)).dropna().astype(str)))
        dlang = txt_langs[0] if len(txt_langs) == 1 else "mixed"

    return method, qlang, dlang


def coalesce_metadata(merged: pd.DataFrame) -> pd.DataFrame:
    for col in ["query_id_base", "split", "category", "question_type3", "selected_role", "query_lang"]:
        ca, cb = f"{col}_tri", f"{col}_txt"
        if ca in merged.columns and cb in merged.columns:
            merged[col] = merged[ca].combine_first(merged[cb])
            merged = merged.drop(columns=[ca, cb])
        elif ca in merged.columns:
            merged = merged.rename(columns={ca: col})
        elif cb in merged.columns:
            merged = merged.rename(columns={cb: col})
    return merged


def weighted_rrf(tri: pd.DataFrame, txt: pd.DataFrame, wt: float, wx: float, k: int, top_k: int) -> pd.DataFrame:
    meta_cols = ["qid", "docno", "rank", "query_id_base", "split", "category", "question_type3", "selected_role", "query_lang"]
    tri_cols = [c for c in meta_cols if c in tri.columns]
    txt_cols = [c for c in meta_cols if c in txt.columns]
    m = tri[tri_cols].merge(txt[txt_cols], on=["qid", "docno"], how="outer", suffixes=("_tri", "_txt"))
    m = coalesce_metadata(m)
    m["rank_tri"] = m["rank_tri"].fillna(10**9)
    m["rank_txt"] = m["rank_txt"].fillna(10**9)
    m["score"] = wt / (k + m["rank_tri"]) + wx / (k + m["rank_txt"])
    m = m.sort_values(["qid", "score"], ascending=[True, False])
    m["rank"] = m.groupby("qid").cumcount() + 1
    return m[m["rank"] <= top_k].copy()


def minmax_by_qid(df: pd.DataFrame, score_col: str) -> pd.Series:
    g = df.groupby("qid")[score_col]
    mn = g.transform("min")
    mx = g.transform("max")
    denom = (mx - mn).replace(0, np.nan)
    return ((df[score_col] - mn) / denom).fillna(0.0)


def weighted_score(tri: pd.DataFrame, txt: pd.DataFrame, wt: float, wx: float, top_k: int) -> pd.DataFrame:
    tri = tri.copy(); txt = txt.copy()
    tri["score_norm"] = minmax_by_qid(tri, "score")
    txt["score_norm"] = minmax_by_qid(txt, "score")
    meta_cols = ["qid", "docno", "score_norm", "query_id_base", "split", "category", "question_type3", "selected_role", "query_lang"]
    tri_cols = [c for c in meta_cols if c in tri.columns]
    txt_cols = [c for c in meta_cols if c in txt.columns]
    m = tri[tri_cols].merge(txt[txt_cols], on=["qid", "docno"], how="outer", suffixes=("_tri", "_txt"))
    m = coalesce_metadata(m)
    m["score_norm_tri"] = m["score_norm_tri"].fillna(0.0)
    m["score_norm_txt"] = m["score_norm_txt"].fillna(0.0)
    m["score"] = wt * m["score_norm_tri"] + wx * m["score_norm_txt"]
    m = m.sort_values(["qid", "score"], ascending=[True, False])
    m["rank"] = m.groupby("qid").cumcount() + 1
    return m[m["rank"] <= top_k].copy()


def union_candidates(tri: pd.DataFrame, txt: pd.DataFrame, top_k: int) -> pd.DataFrame:
    both = pd.concat([tri, txt], ignore_index=True)
    both = both.sort_values(["qid", "rank"])
    meta_cols = ["qid", "docno", "query_id_base", "split", "category", "question_type3", "selected_role", "query_lang"]
    meta_cols = [c for c in meta_cols if c in both.columns]
    out = both[meta_cols + ["score"]].drop_duplicates(["qid", "docno"], keep="first")
    out = out.sort_values(["qid", "score"], ascending=[True, False])
    out["rank"] = out.groupby("qid").cumcount() + 1
    return out[out["rank"] <= top_k].copy()


def main() -> None:
    args = parse_args()
    profiler = ResourceProfiler(enabled=args.profile)
    profiler.start()

    top_for_union = args.candidate_top_n if args.fusion == "union" else None

    profiler.start_stage("load_runs")
    tri = load_run(args.run_triples, "triples", top_for_union)
    txt = load_run(args.run_text, "text", top_for_union)
    profiler.stop_stage("load_runs")

    method_base, qlang, dlang = infer_meta(tri, txt, args.method, args.query_lang, args.doc_lang)

    profiler.start_stage("fusion")
    if args.fusion == "rrf":
        run = weighted_rrf(tri, txt, args.weight_triples, args.weight_text, args.rrf_k, args.top_k)
        repr_name = "hybrid_rrf"
    elif args.fusion == "score":
        run = weighted_score(tri, txt, args.weight_triples, args.weight_text, args.top_k)
        repr_name = "hybrid_score"
    else:
        run = union_candidates(tri, txt, args.top_k if args.top_k else args.candidate_top_n)
        repr_name = "hybrid_union_candidates"
    profiler.stop_stage("fusion")

    profiler.start_stage("write")
    save_run(
        run,
        args.output,
        method=method_base,
        representation=repr_name,
        query_lang=qlang,
        doc_lang=dlang,
        extra={
            "fusion": args.fusion,
            "weight_triples": args.weight_triples,
            "weight_text": args.weight_text,
            "rrf_k": args.rrf_k,
        },
    )
    profiler.stop_stage("write")

    profiler.stop()
    if args.profile:
        summary = profiler.summary()
        total = summary.get("total_time_s")
        fusion_time = summary.get("fusion_time_s")
        n_q = int(run["qid"].nunique()) if len(run) else 0
        profiler.write_json(
            args.profile_path or profile_path_from_output(args.output),
            extra={
                "method": method_base,
                "representation": repr_name,
                "query_lang": qlang,
                "doc_lang": dlang,
                "top_k": args.top_k,
                "output": str(args.output),
                "fusion": args.fusion,
                "run_triples": args.run_triples,
                "run_text": args.run_text,
                "n_triples_rows_in": int(len(tri)),
                "n_text_rows_in": int(len(txt)),
                "n_queries": n_q,
                "n_results": int(len(run)),
                "queries_per_second_total": safe_rate(n_q, total),
                "fusion_ms_per_query": safe_ms_per_item(fusion_time, n_q),
            },
        )


if __name__ == "__main__":
    main()
