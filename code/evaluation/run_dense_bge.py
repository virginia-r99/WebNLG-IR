#!/usr/bin/env python3
"""Dense BGE-M3 retrieval over WebNLG canonical document representations.

GPU/batching notes
------------------
This script uses batched encoding and chunked query scoring. Main speed controls:
  --doc_batch_size       batch size for document embedding
  --query_batch_size     batch size for query embedding
  --query_chunk_size     number of query embeddings scored against all docs at once
  --device cuda          use GPU when available

If CUDA runs out of memory, lower --doc_batch_size and/or --query_chunk_size.
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from ir_common import (
    attach_query_metadata,
    effective_doc_lang,
    iter_batches,
    load_docs_for_run,
    load_queries_for_run,
    save_run,
    topk_from_scores,
)
from resource_profiler import ResourceProfiler, profile_path_from_output, safe_ms_per_item, safe_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--query_lang", choices=["en", "ca", "es"], required=True)
    p.add_argument("--doc_lang", choices=["en", "es", "mix"], required=True, help="Document language/condition for triples/text/concat: en, es, or mix.")
    p.add_argument("--representation", choices=["triples", "text", "concat"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="ALL")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--model_name", default="BAAI/bge-m3")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--doc_batch_size", type=int, default=128)
    p.add_argument("--query_batch_size", type=int, default=128)
    p.add_argument("--query_chunk_size", type=int, default=2048, help="Number of queries scored against all documents at once.")
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu, etc. Defaults to cuda if available.")
    p.add_argument("--no_fp16", action="store_true", help="Disable fp16 model loading on CUDA.")
    p.add_argument("--profile", action="store_true", help="Write a .profile.json resource log next to the run file.")
    p.add_argument("--profile_path", default=None, help="Optional explicit path for the resource profile JSON.")
    return p.parse_args()


def build_model(model_name: str, max_seq_length: int, device: str | None, no_fp16: bool):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"device": device}
    if str(device).startswith("cuda") and not no_fp16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_seq_length
    model.eval()
    return model, device


def encode(model, texts: list[str], is_query: bool, batch_size: int) -> np.ndarray:
    prefix = "query: " if is_query else "passage: "
    texts = [prefix + str(x) for x in texts]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


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
    model, device = build_model(args.model_name, args.max_seq_length, args.device, args.no_fp16)
    profiler.stop_stage("model_load")

    print(
        "BGE dense: "
        f"query_lang={args.query_lang}, doc_lang={doc_lang}, "
        f"representation={args.representation}, device={device}, "
        f"docs={len(docs):,}, queries={len(queries):,}, "
        f"doc_batch_size={args.doc_batch_size}, query_batch_size={args.query_batch_size}, "
        f"query_chunk_size={args.query_chunk_size}"
    )

    profiler.start_stage("doc_encode")
    doc_emb = encode(model, docs["text"].tolist(), is_query=False, batch_size=args.doc_batch_size)
    profiler.stop_stage("doc_encode")
    docnos = docs["docno"].astype(str).to_numpy()

    runs = []
    profiler.start_stage("query_encode_and_scoring")
    total_query_encode_time = 0.0
    total_scoring_time = 0.0
    import time
    for _, q_chunk in iter_batches(queries, args.query_chunk_size):
        t0 = time.perf_counter()
        q_emb = encode(model, q_chunk["query"].tolist(), is_query=True, batch_size=args.query_batch_size)
        total_query_encode_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = q_emb @ doc_emb.T
        runs.append(topk_from_scores(q_chunk["qid"].astype(str).to_numpy(), docnos, scores, args.top_k))
        total_scoring_time += time.perf_counter() - t0
        del q_emb, scores
        gc.collect()
    profiler.stop_stage("query_encode_and_scoring")
    profiler.stage_times["query_encode"] = profiler.stage_times.get("query_encode", 0.0) + total_query_encode_time
    profiler.stage_times["scoring"] = profiler.stage_times.get("scoring", 0.0) + total_scoring_time

    run = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    profiler.start_stage("metadata_attach")
    run = attach_query_metadata(run, queries)
    profiler.stop_stage("metadata_attach")

    profiler.start_stage("write")
    save_run(
        run,
        args.output,
        method="bge_m3_dense",
        representation=args.representation,
        query_lang=args.query_lang,
        doc_lang=doc_lang,
        extra={
            "model_name": args.model_name,
            "doc_batch_size": args.doc_batch_size,
            "query_batch_size": args.query_batch_size,
            "query_chunk_size": args.query_chunk_size,
            "device": str(device),
        },
    )
    profiler.stop_stage("write")

    profiler.stop()
    if args.profile:
        summary = profiler.summary()
        total = summary.get("total_time_s")
        doc_encode_time = summary.get("doc_encode_time_s")
        query_encode_time = summary.get("query_encode_time_s")
        scoring_time = summary.get("scoring_time_s")
        profiler.write_json(
            args.profile_path or profile_path_from_output(args.output),
            extra={
                "method": "bge_m3_dense",
                "representation": args.representation,
                "query_lang": args.query_lang,
                "doc_lang": doc_lang,
                "split": args.split,
                "top_k": args.top_k,
                "output": str(args.output),
                "model_name": args.model_name,
                "device": str(device),
                "max_seq_length": args.max_seq_length,
                "doc_batch_size": args.doc_batch_size,
                "query_batch_size": args.query_batch_size,
                "query_chunk_size": args.query_chunk_size,
                "n_docs": int(len(docs)),
                "n_queries": int(len(queries)),
                "n_results": int(len(run)),
                "docs_per_second_total": safe_rate(len(docs), total),
                "queries_per_second_total": safe_rate(len(queries), total),
                "doc_encode_ms_per_doc": safe_ms_per_item(doc_encode_time, len(docs)),
                "query_encode_ms_per_query": safe_ms_per_item(query_encode_time, len(queries)),
                "scoring_ms_per_query": safe_ms_per_item(scoring_time, len(queries)),
            },
        )


if __name__ == "__main__":
    main()
