#!/usr/bin/env python3
"""MILCO learned-sparse retrieval over WebNLG canonical document representations.

GPU/batching notes
------------------
MILCO encoding can be expensive. This script batches both document and query
encoding:
  --doc_batch_size       batch size for document sparse encoding
  --query_batch_size     batch size for query sparse encoding
  --query_chunk_size     number of encoded queries scored against the doc matrix
  --device cuda          use GPU when available

The scoring itself is sparse matrix multiplication on CPU after vectors are
converted to scipy CSR. The GPU speedup comes from model encoding.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import gc
import time

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from ir_common import (
    attach_query_metadata,
    effective_doc_lang,
    iter_batches,
    load_docs_for_run,
    load_queries_for_run,
    save_run,
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
    p.add_argument("--model_name", default="omai-research/milco-650m")
    p.add_argument("--doc_batch_size", type=int, default=64)
    p.add_argument("--query_batch_size", type=int, default=64)
    p.add_argument("--query_chunk_size", type=int, default=512)
    p.add_argument("--source_view", action="store_true", default=True)
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu, etc. Defaults to cuda if available.")
    p.add_argument("--profile", action="store_true", help="Write a .profile.json resource log next to the run file.")
    p.add_argument("--profile_path", default=None, help="Optional explicit path for the resource profile JSON.")
    return p.parse_args()


def to_csr(x) -> sparse.csr_matrix:
    import torch

    if sparse.issparse(x):
        return x.tocsr()

    if torch.is_tensor(x):
        x = x.detach().cpu().to(torch.float32)
        if x.is_sparse:
            x = x.coalesce()
            idx = x.indices().numpy()
            vals = x.values().numpy()
            return sparse.csr_matrix((vals, (idx[0], idx[1])), shape=x.shape)
        arr = x.numpy()
        return sparse.csr_matrix(arr)

    if isinstance(x, np.ndarray):
        return sparse.csr_matrix(x)

    raise TypeError(f"Cannot convert {type(x)} to CSR")


def load_model(model_name: str, device: str | None):
    from transformers import AutoModel
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, device


def encode_docs(model, texts: list[str], source_view: bool, batch_size: int) -> sparse.csr_matrix:
    mats = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding MILCO docs"):
        batch = [str(x) for x in texts[start:start + batch_size]]
        mats.append(to_csr(model.encode_document(batch, source_view=source_view)))
    return sparse.vstack(mats).tocsr() if mats else sparse.csr_matrix((0, 0))


def encode_queries(model, texts: list[str], source_view: bool, batch_size: int) -> sparse.csr_matrix:
    mats = []
    for start in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[start:start + batch_size]]
        mats.append(to_csr(model.encode_query(batch, source_view=source_view)))
    return sparse.vstack(mats).tocsr() if mats else sparse.csr_matrix((0, 0))


def topk_sparse_scores(qids: np.ndarray, docnos: np.ndarray, sims: np.ndarray, top_k: int) -> pd.DataFrame:
    n_q, n_d = sims.shape
    k = min(top_k, n_d)
    if k <= 0:
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    row = np.arange(n_q)[:, None]
    top_scores = sims[row, idx]
    order = np.argsort(-top_scores, axis=1)
    idx = idx[row, order]
    top_scores = top_scores[row, order]

    return pd.DataFrame({
        "qid": np.repeat(qids, k),
        "docno": docnos[idx.reshape(-1)],
        "score": top_scores.reshape(-1),
        "rank": np.tile(np.arange(1, k + 1), n_q),
    })


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
    model, device = load_model(args.model_name, args.device)
    profiler.stop_stage("model_load")

    print(
        "MILCO: "
        f"query_lang={args.query_lang}, doc_lang={doc_lang}, "
        f"representation={args.representation}, device={device}, "
        f"docs={len(docs):,}, queries={len(queries):,}, "
        f"doc_batch_size={args.doc_batch_size}, query_batch_size={args.query_batch_size}, "
        f"query_chunk_size={args.query_chunk_size}"
    )

    profiler.start_stage("doc_encode")
    doc_matrix = encode_docs(model, docs["text"].tolist(), args.source_view, args.doc_batch_size)
    profiler.stop_stage("doc_encode")
    docnos = docs["docno"].astype(str).to_numpy()

    runs = []
    total_query_encode_time = 0.0
    total_scoring_time = 0.0
    profiler.start_stage("query_encode_and_scoring")
    for _, q_chunk in iter_batches(queries, args.query_chunk_size):
        t0 = time.perf_counter()
        q_matrix = encode_queries(model, q_chunk["query"].tolist(), args.source_view, args.query_batch_size)
        total_query_encode_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        sims = q_matrix.dot(doc_matrix.T).toarray()
        runs.append(topk_sparse_scores(q_chunk["qid"].astype(str).to_numpy(), docnos, sims, args.top_k))
        total_scoring_time += time.perf_counter() - t0
        del q_matrix, sims
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
        method="milco_sparse",
        representation=args.representation,
        query_lang=args.query_lang,
        doc_lang=doc_lang,
        extra={
            "model_name": args.model_name,
            "source_view": args.source_view,
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
                "method": "milco_sparse",
                "representation": args.representation,
                "query_lang": args.query_lang,
                "doc_lang": doc_lang,
                "split": args.split,
                "top_k": args.top_k,
                "output": str(args.output),
                "model_name": args.model_name,
                "source_view": args.source_view,
                "device": str(device),
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
                "doc_matrix_shape": list(doc_matrix.shape),
                "doc_matrix_nnz": int(doc_matrix.nnz),
            },
        )


if __name__ == "__main__":
    main()
