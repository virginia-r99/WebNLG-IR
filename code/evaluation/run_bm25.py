#!/usr/bin/env python3
"""Pure-Python/NumPy BM25 for WebNLG canonical document runs."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ir_common import attach_query_metadata, effective_doc_lang, load_docs_for_run, load_queries_for_run, save_run, tokenize
from resource_profiler import ResourceProfiler, profile_path_from_output, safe_ms_per_item, safe_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--query_lang", choices=["en", "ca", "es"], required=True)
    p.add_argument("--doc_lang", choices=["en", "es", "mix"], required=True,
                   help="Document language/condition for triples/text/concat: en, es, or mix.")
    p.add_argument("--representation", choices=["triples", "text", "concat"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="ALL", help="Query split to run: train/dev/test/ALL. Default ALL uses all questions.")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--k1", type=float, default=0.9)
    p.add_argument("--b", type=float, default=0.4)
    p.add_argument("--profile", action="store_true", help="Write a .profile.json resource log next to the run file.")
    p.add_argument("--profile_path", default=None, help="Optional explicit path for the resource profile JSON.")
    return p.parse_args()


class BM25Index:
    def __init__(self, docs: pd.DataFrame, k1: float = 0.9, b: float = 0.4):
        self.docs = docs.reset_index(drop=True)
        self.docnos = self.docs["docno"].astype(str).to_numpy()
        self.k1 = k1
        self.b = b
        self.N = len(self.docs)
        self.doc_len = np.zeros(self.N, dtype=np.float32)
        self.avgdl = 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in tqdm(enumerate(self.docs["text"].tolist()), total=self.N, desc="Indexing BM25"):
            counts = Counter(tokenize(text))
            self.doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                tmp[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        for term, pairs in tmp.items():
            df = len(pairs)
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            doc_ids = np.fromiter((p[0] for p in pairs), dtype=np.int32)
            tfs = np.fromiter((p[1] for p in pairs), dtype=np.float32)
            self.postings[term] = (doc_ids, tfs)

    def search_one(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(self.N, dtype=np.float32)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            doc_ids, tfs = self.postings[term]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_ids] / max(self.avgdl, 1e-9))
            scores[doc_ids] += self.idf[term] * (tfs * (self.k1 + 1.0)) / denom
        k = min(top_k, self.N)
        if k == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def search(self, queries: pd.DataFrame, top_k: int) -> pd.DataFrame:
        rows = []
        for _, q in tqdm(queries.iterrows(), total=len(queries), desc="BM25 search"):
            idx, scores = self.search_one(q["query"], top_k)
            for rank, (di, score) in enumerate(zip(idx, scores), start=1):
                rows.append((q["qid"], self.docnos[di], float(score), rank))
        return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])


def main() -> None:
    args = parse_args()
    profiler = ResourceProfiler(enabled=args.profile)
    profiler.start()

    doc_lang = effective_doc_lang(args.representation, args.doc_lang)

    profiler.start_stage("load_data")
    docs = load_docs_for_run(args.data_dir, args.representation, args.doc_lang)
    queries = load_queries_for_run(args.data_dir, args.query_lang, args.split)
    profiler.stop_stage("load_data")

    print(f"BM25: query_lang={args.query_lang}, doc_lang={doc_lang}, representation={args.representation}, queries={len(queries):,}, docs={len(docs):,}")

    profiler.start_stage("index")
    index = BM25Index(docs, k1=args.k1, b=args.b)
    profiler.stop_stage("index")

    profiler.start_stage("retrieval")
    run = index.search(queries, args.top_k)
    profiler.stop_stage("retrieval")

    profiler.start_stage("metadata_attach")
    run = attach_query_metadata(run, queries)
    profiler.stop_stage("metadata_attach")

    profiler.start_stage("write")
    save_run(
        run,
        args.output,
        method="bm25",
        representation=args.representation,
        query_lang=args.query_lang,
        doc_lang=doc_lang,
        extra={"bm25_k1": args.k1, "bm25_b": args.b},
    )
    profiler.stop_stage("write")

    profiler.stop()
    if args.profile:
        total = profiler.summary().get("total_time_s")
        index_time = profiler.summary().get("index_time_s")
        retrieval_time = profiler.summary().get("retrieval_time_s")
        profiler.write_json(
            args.profile_path or profile_path_from_output(args.output),
            extra={
                "method": "bm25",
                "representation": args.representation,
                "query_lang": args.query_lang,
                "doc_lang": doc_lang,
                "split": args.split,
                "top_k": args.top_k,
                "output": str(args.output),
                "n_docs": int(len(docs)),
                "n_queries": int(len(queries)),
                "n_results": int(len(run)),
                "k1": args.k1,
                "b": args.b,
                "docs_per_second_total": safe_rate(len(docs), total),
                "queries_per_second_total": safe_rate(len(queries), total),
                "index_ms_per_doc": safe_ms_per_item(index_time, len(docs)),
                "retrieval_ms_per_query": safe_ms_per_item(retrieval_time, len(queries)),
            },
        )


if __name__ == "__main__":
    main()
