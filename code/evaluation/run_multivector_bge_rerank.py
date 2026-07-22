#!/usr/bin/env python3
"""BGE-M3 multi-vector late-interaction reranking for WebNLG IR.

Recommended use
---------------
1) Run dense BGE-M3 for triples/text/concat with top_k=1000.
2) Use this script to rerank those candidates with BGE-M3 ColBERT vectors.

For the refined hybrid cascade, pass a union candidate run produced by:
    run_hybrid_fusion.py --fusion union

GPU/batching notes
------------------
This version batches:
  - document ColBERT-vector encoding: --doc_batch_size
  - query ColBERT-vector encoding:    --query_batch_size
  - candidate MaxSim scoring:         --rerank_batch_size
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import gc
import time
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_common import (
    attach_query_metadata,
    effective_doc_lang,
    load_docs_for_run,
    load_queries_for_run,
    read_table,
    save_run,
)
from resource_profiler import ResourceProfiler, profile_path_from_output, safe_ms_per_item, safe_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--query_lang", choices=["en", "ca", "es"], required=True)
    p.add_argument("--doc_lang", choices=["en", "es", "mix"], required=True, help="Document language/condition for triples/text/concat: en, es, or mix.")
    p.add_argument("--representation", choices=["triples", "text", "concat"], required=True, help="Representation text used for multi-vector scoring.")
    p.add_argument("--output", required=True)
    p.add_argument("--candidate_run", default=None, help="Run file with qid/docno candidates. If omitted, use --full_index to score all docs.")
    p.add_argument("--full_index", action="store_true", help="Score every document for every query. Feasible only for small query subsets.")
    p.add_argument("--candidate_top_n", type=int, default=1000)
    p.add_argument("--split", default="ALL")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--model_name", default="BAAI/bge-m3")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--doc_batch_size", type=int, default=64)
    p.add_argument("--query_batch_size", type=int, default=32)
    p.add_argument("--rerank_batch_size", type=int, default=64, help="Number of candidate documents scored together per query on the selected device.")
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu, etc. Defaults to cuda if available.")
    p.add_argument("--no_fp16", action="store_true", help="Disable fp16 model/scoring on CUDA.")
    p.add_argument("--profile", action="store_true", help="Write a .profile.json resource log next to the run file.")
    p.add_argument("--profile_path", default=None, help="Optional explicit path for the resource profile JSON.")
    return p.parse_args()


def load_model(model_name: str, device: str | None, no_fp16: bool):
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = str(device).startswith("cuda") and not no_fp16
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
    return model, device, use_fp16


def encode_colbert(model, texts: list[str], batch_size: int, max_length: int, desc: str) -> list[np.ndarray]:
    vecs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = [str(x) for x in texts[start:start + batch_size]]
        out = model.encode(
            batch,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        cv = out["colbert_vecs"]
        if isinstance(cv, np.ndarray) and cv.dtype != object and cv.ndim == 3:
            vecs.extend([np.asarray(cv[i], dtype=np.float32) for i in range(cv.shape[0])])
        else:
            vecs.extend([np.asarray(x, dtype=np.float32) for x in cv])
    return vecs


def chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def maxsim_scores_batched(q_vec: np.ndarray, doc_vec_batch: list[np.ndarray], device: str, use_fp16: bool) -> np.ndarray:
    """Compute ColBERT MaxSim scores for one query against a batch of docs."""
    import torch

    if q_vec.size == 0 or not doc_vec_batch:
        return np.zeros(len(doc_vec_batch), dtype=np.float32)

    dtype = torch.float16 if str(device).startswith("cuda") and use_fp16 else torch.float32
    q = torch.as_tensor(q_vec, dtype=dtype, device=device)
    if q.ndim != 2 or q.shape[0] == 0:
        return np.zeros(len(doc_vec_batch), dtype=np.float32)

    dim = int(q.shape[1])
    lengths = [int(d.shape[0]) if d is not None and d.size > 0 else 0 for d in doc_vec_batch]
    max_len = max(lengths) if lengths else 0
    if max_len <= 0:
        return np.zeros(len(doc_vec_batch), dtype=np.float32)

    docs = torch.zeros((len(doc_vec_batch), max_len, dim), dtype=dtype, device=device)
    mask = torch.zeros((len(doc_vec_batch), max_len), dtype=torch.bool, device=device)

    for i, d in enumerate(doc_vec_batch):
        if d is None or d.size == 0:
            continue
        d_arr = np.asarray(d, dtype=np.float32)
        if d_arr.ndim != 2 or d_arr.shape[1] != dim:
            continue
        L = min(d_arr.shape[0], max_len)
        docs[i, :L, :] = torch.as_tensor(d_arr[:L], dtype=dtype, device=device)
        mask[i, :L] = True

    sims = torch.einsum("qd,bld->bql", q, docs)
    neg_large = -torch.finfo(dtype).max
    sims = sims.masked_fill(~mask[:, None, :], neg_large)
    scores = sims.max(dim=2).values.sum(dim=1)
    return scores.detach().float().cpu().numpy()


def build_candidate_map(candidate_run: str | None, queries: pd.DataFrame, docs: pd.DataFrame, candidate_top_n: int, full_index: bool) -> dict[str, list[str]]:
    if candidate_run:
        cand = read_table(candidate_run)
        cand["qid"] = cand["qid"].astype(str)
        cand["docno"] = cand["docno"].astype(str)
        cand = cand[cand["qid"].isin(set(queries["qid"].astype(str)))].copy()
        if "rank" in cand.columns:
            cand = cand.sort_values(["qid", "rank"])
            cand = cand.groupby("qid", as_index=False).head(candidate_top_n)
        else:
            cand = cand.drop_duplicates(["qid", "docno"]).groupby("qid", as_index=False).head(candidate_top_n)
        return cand.groupby("qid")["docno"].apply(lambda x: list(dict.fromkeys(map(str, x)))).to_dict()

    if not full_index:
        raise ValueError("Pass --candidate_run or explicitly set --full_index.")

    all_docnos = docs["docno"].astype(str).tolist()
    return {str(qid): all_docnos for qid in queries["qid"].astype(str)}


def main() -> None:
    args = parse_args()
    profiler = ResourceProfiler(enabled=args.profile, device=args.device)
    profiler.start()

    doc_lang = effective_doc_lang(args.representation, args.doc_lang)

    profiler.start_stage("load_data")
    docs = load_docs_for_run(args.data_dir, args.representation, args.doc_lang)
    queries = load_queries_for_run(args.data_dir, args.query_lang, args.split)
    profiler.stop_stage("load_data")

    profiler.start_stage("model_load")
    model, device, use_fp16 = load_model(args.model_name, args.device, args.no_fp16)
    profiler.stop_stage("model_load")

    print(
        "BGE multivector: "
        f"query_lang={args.query_lang}, doc_lang={doc_lang}, "
        f"representation={args.representation}, device={device}, use_fp16={use_fp16}, "
        f"docs={len(docs):,}, queries={len(queries):,}, "
        f"doc_batch_size={args.doc_batch_size}, query_batch_size={args.query_batch_size}, "
        f"rerank_batch_size={args.rerank_batch_size}, candidate_top_n={args.candidate_top_n}"
    )

    profiler.start_stage("load_candidates")
    candidate_map = build_candidate_map(args.candidate_run, queries, docs, args.candidate_top_n, args.full_index)
    n_candidate_pairs = int(sum(len(v) for v in candidate_map.values()))
    profiler.stop_stage("load_candidates")

    profiler.start_stage("doc_encode")
    doc_vecs = encode_colbert(model, docs["text"].tolist(), args.doc_batch_size, args.max_seq_length, "Encoding MV docs")
    profiler.stop_stage("doc_encode")
    docno_to_idx = {str(d): i for i, d in enumerate(docs["docno"].astype(str))}

    rows = []
    total_query_encode_time = 0.0
    total_rerank_time = 0.0
    profiler.start_stage("query_encode_and_rerank")
    for start in tqdm(range(0, len(queries), args.query_batch_size), desc="Reranking query batches"):
        q_chunk = queries.iloc[start:start + args.query_batch_size]
        t0 = time.perf_counter()
        q_vecs = encode_colbert(model, q_chunk["query"].tolist(), args.query_batch_size, args.max_seq_length, "Encoding MV queries")
        total_query_encode_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        for row_i, (_, qrow) in enumerate(q_chunk.iterrows()):
            qid = str(qrow["qid"])
            cands = [d for d in candidate_map.get(qid, []) if d in docno_to_idx]
            scored: list[tuple[str, float]] = []
            for cand_batch in chunks(cands, args.rerank_batch_size):
                batch_vecs = [doc_vecs[docno_to_idx[docno]] for docno in cand_batch]
                batch_scores = maxsim_scores_batched(q_vecs[row_i], batch_vecs, str(device), use_fp16)
                scored.extend((docno, float(score)) for docno, score in zip(cand_batch, batch_scores))

            scored.sort(key=lambda x: x[1], reverse=True)
            for rank, (docno, score) in enumerate(scored[:args.top_k], start=1):
                rows.append((qid, docno, score, rank))
        total_rerank_time += time.perf_counter() - t0
        del q_vecs
        gc.collect()
    profiler.stop_stage("query_encode_and_rerank")
    profiler.stage_times["query_encode"] = profiler.stage_times.get("query_encode", 0.0) + total_query_encode_time
    profiler.stage_times["rerank"] = profiler.stage_times.get("rerank", 0.0) + total_rerank_time

    run = pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])

    profiler.start_stage("metadata_attach")
    run = attach_query_metadata(run, queries)
    profiler.stop_stage("metadata_attach")

    profiler.start_stage("write")
    save_run(
        run,
        args.output,
        method="bge_m3_multivector",
        representation=args.representation,
        query_lang=args.query_lang,
        doc_lang=doc_lang,
        extra={
            "model_name": args.model_name,
            "candidate_top_n": args.candidate_top_n,
            "candidate_run": args.candidate_run or "FULL_INDEX",
            "doc_batch_size": args.doc_batch_size,
            "query_batch_size": args.query_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "device": str(device),
            "use_fp16": use_fp16,
        },
    )
    profiler.stop_stage("write")

    profiler.stop()
    if args.profile:
        summary = profiler.summary()
        total = summary.get("total_time_s")
        doc_encode_time = summary.get("doc_encode_time_s")
        query_encode_time = summary.get("query_encode_time_s")
        rerank_time = summary.get("rerank_time_s")
        profiler.write_json(
            args.profile_path or profile_path_from_output(args.output),
            extra={
                "method": "bge_m3_multivector",
                "representation": args.representation,
                "query_lang": args.query_lang,
                "doc_lang": doc_lang,
                "split": args.split,
                "top_k": args.top_k,
                "output": str(args.output),
                "model_name": args.model_name,
                "candidate_top_n": args.candidate_top_n,
                "candidate_run": args.candidate_run or "FULL_INDEX",
                "device": str(device),
                "use_fp16": use_fp16,
                "max_seq_length": args.max_seq_length,
                "doc_batch_size": args.doc_batch_size,
                "query_batch_size": args.query_batch_size,
                "rerank_batch_size": args.rerank_batch_size,
                "n_docs": int(len(docs)),
                "n_queries": int(len(queries)),
                "n_candidate_pairs": n_candidate_pairs,
                "n_results": int(len(run)),
                "docs_per_second_total": safe_rate(len(docs), total),
                "queries_per_second_total": safe_rate(len(queries), total),
                "doc_encode_ms_per_doc": safe_ms_per_item(doc_encode_time, len(docs)),
                "query_encode_ms_per_query": safe_ms_per_item(query_encode_time, len(queries)),
                "rerank_ms_per_candidate_pair": safe_ms_per_item(rerank_time, n_candidate_pairs),
            },
        )


if __name__ == "__main__":
    main()
