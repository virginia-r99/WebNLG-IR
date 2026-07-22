#!/usr/bin/env python3
"""Print shell commands for the WebNLG lang3 IR experimental matrix.

Lang3 matrix
------------
- Query languages: en, es.
- Triples are evaluated for document conditions: en, es, mix.
- Text-only is evaluated for real lexicalisation languages: en, es by default.
  text_mix exists in prepared docs for concat/reranking compatibility, but it is
  intentionally not part of the main text-only matrix because it duplicates EN text.
- Concat is evaluated for: en, es, mix.
  concat_mix = triples_mix + text_mix, where text_mix is the English verbalisation.
- Hybrids are paired within their document condition:
    en  => triples_en  + text_en
    es  => triples_es  + text_es
    mix => triples_mix + text_en by default, output labelled doc_lang=mix.
  This avoids cross-language hybrids such as triples_es + text_en unless explicitly
  requested by changing --mix_text_lang or editing the script.

  Call:
  python make_matrix_commands.py \
  --data_dir ./prepared_min3_mixed_normtext \
  --results_dir ./runs_min3_mixed_normtext \
  --eval_dir ./eval_min3_mixed_normtext \
  --split ALL \
  --top_k_base 100 \
  --top_k_final 10 \
  --device cuda \
  --doc_batch_size 128 \
  --query_batch_size 128 \
  --query_chunk_size 1024 \
  --mv_doc_batch_size 64 \
  --mv_query_batch_size 32 \
  --mv_rerank_batch_size 64 \
  --ks 5 10 \
  --profile \
  > run_matrix_min3_mixed_normtext.sh
  
  bash run_matrix_min3_mixed_normtext.sh
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

QUERY_LANG_CHOICES = ["en", "ca", "es"]
DOC_LANG_CHOICES = ["en", "es", "mix"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./prepared_lang3_min2")
    p.add_argument("--results_dir", default="./runs_lang3_min2")
    p.add_argument("--eval_dir", default="./eval_lang3_min2")
    p.add_argument("--split", default="ALL", help="Default ALL uses every available query, not just test.")
    p.add_argument("--top_k_base", type=int, default=1000, help="Use >=1000 if reranking with multi-vector.")
    p.add_argument("--top_k_final", type=int, default=100)
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 100], help="Evaluation cutoffs.")
    p.add_argument("--profile", action="store_true", help="Write .profile.json logs and aggregate them.")

    p.add_argument("--include_milco", action="store_true", default=True)
    p.add_argument("--no_milco", dest="include_milco", action="store_false")
    p.add_argument("--include_multivector", action="store_true", default=True)
    p.add_argument("--no_multivector", dest="include_multivector", action="store_false")
    p.add_argument("--ext", choices=["parquet", "csv"], default="parquet")

    p.add_argument("--query_langs", nargs="+", choices=QUERY_LANG_CHOICES, default=["en", "ca", "es"])

    # Global doc_langs controls triples/concat/hybrid unless the more specific
    # options below are passed.
    p.add_argument("--doc_langs", nargs="+", choices=DOC_LANG_CHOICES, default=["en", "es", "mix"])
    p.add_argument("--triple_doc_langs", nargs="+", choices=DOC_LANG_CHOICES, default=None)
    p.add_argument("--text_doc_langs", nargs="+", choices=DOC_LANG_CHOICES, default=["en", "es"])
    p.add_argument("--concat_doc_langs", nargs="+", choices=DOC_LANG_CHOICES, default=None)
    p.add_argument("--hybrid_doc_langs", nargs="+", choices=DOC_LANG_CHOICES, default=None)
    p.add_argument(
        "--mix_text_lang",
        choices=["en", "es", "mix"],
        default="en",
        help="Text run used when fusing/reranking doc_lang=mix. Default en because mix text is English verbalisation.",
    )

    # Shared neural batching/device options.
    p.add_argument("--device", default=None, help="cuda, cuda:0, cpu. If omitted, neural scripts auto-detect.")
    p.add_argument("--doc_batch_size", type=int, default=128)
    p.add_argument("--query_batch_size", type=int, default=128)
    p.add_argument("--query_chunk_size", type=int, default=1024)

    # Multi-vector-specific options.
    p.add_argument("--mv_doc_batch_size", type=int, default=64)
    p.add_argument("--mv_query_batch_size", type=int, default=32)
    p.add_argument("--mv_rerank_batch_size", type=int, default=64)
    p.add_argument("--mv_max_seq_length", type=int, default=512)

    # Optional model names.
    p.add_argument("--dense_model_name", default="BAAI/bge-m3")
    p.add_argument("--milco_model_name", default="omai-research/milco-650m")
    p.add_argument("--mv_model_name", default="BAAI/bge-m3")

    return p.parse_args()


def maybe_device_arg(args) -> str:
    return f" --device {args.device}" if args.device else ""


def uniq_keep_order(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def main():
    args = parse_args()
    R = Path(args.results_dir)
    print("set -euo pipefail")
    print(f"mkdir -p {R}")

    base_methods = [
        ("bm25", "python run_bm25.py"),
        ("bge_m3_dense", "python run_dense_bge.py"),
    ]
    if args.include_milco:
        base_methods.append(("milco_sparse", "python run_sparse_milco.py"))

    query_langs = args.query_langs
    triple_doc_langs = args.triple_doc_langs or args.doc_langs
    concat_doc_langs = args.concat_doc_langs or args.doc_langs
    hybrid_doc_langs = args.hybrid_doc_langs or args.doc_langs
    text_doc_langs = args.text_doc_langs

    # Text runs needed for the matrix plus any text condition needed by mix hybrids.
    required_text_doc_langs = uniq_keep_order(text_doc_langs + ([args.mix_text_lang] if "mix" in hybrid_doc_langs else []))

    device_arg = maybe_device_arg(args)
    profile_arg = " --profile" if args.profile else ""

    def path(method_name: str, rep: str, qlang: str, dlang: str) -> Path:
        return R / f"{method_name}__{rep}__q{qlang}_d{dlang}.{args.ext}"

    def print_base_run(method_name: str, cmd: str, qlang: str, dlang: str, rep: str):
        topk = args.top_k_base if method_name == "bge_m3_dense" else args.top_k_final
        out = path(method_name, rep, qlang, dlang)

        if method_name == "bge_m3_dense":
            extra = (
                f" --model_name {args.dense_model_name}"
                f" --doc_batch_size {args.doc_batch_size}"
                f" --query_batch_size {args.query_batch_size}"
                f" --query_chunk_size {args.query_chunk_size}"
                f"{device_arg}"
            )
        elif method_name == "milco_sparse":
            extra = (
                f" --model_name {args.milco_model_name}"
                f" --doc_batch_size {args.doc_batch_size}"
                f" --query_batch_size {args.query_batch_size}"
                f" --query_chunk_size {args.query_chunk_size}"
                f"{device_arg}"
            )
        else:
            extra = ""

        print(
            f"{cmd} --data_dir {args.data_dir} --query_lang {qlang} "
            f"--doc_lang {dlang} --representation {rep} --split {args.split} "
            f"--top_k {topk} --output {out}{extra}{profile_arg}"
        )

    # Base retrieval runs.
    for method_name, cmd in base_methods:
        for qlang in query_langs:
            for dlang in triple_doc_langs:
                print_base_run(method_name, cmd, qlang, dlang, "triples")
            for dlang in required_text_doc_langs:
                print_base_run(method_name, cmd, qlang, dlang, "text")
            for dlang in concat_doc_langs:
                print_base_run(method_name, cmd, qlang, dlang, "concat")

    def text_lang_for_hybrid(dlang: str) -> str:
        return args.mix_text_lang if dlang == "mix" else dlang

    # Representation fusion.
    for method_name, _ in base_methods:
        for qlang in query_langs:
            for dlang in hybrid_doc_langs:
                tlang = text_lang_for_hybrid(dlang)
                tri = path(method_name, "triples", qlang, dlang)
                txt = path(method_name, "text", qlang, tlang)
                rrf = path(method_name, "hybrid_rrf", qlang, dlang)
                score = path(method_name, "hybrid_score", qlang, dlang)
                print(
                    f"python run_hybrid_fusion.py --fusion rrf --run_triples {tri} --run_text {txt} "
                    f"--query_lang {qlang} --doc_lang {dlang} --top_k {args.top_k_final} --output {rrf}{profile_arg}"
                )
                print(
                    f"python run_hybrid_fusion.py --fusion score --run_triples {tri} --run_text {txt} "
                    f"--query_lang {qlang} --doc_lang {dlang} --top_k {args.top_k_final} --output {score}{profile_arg}"
                )

    # Multi-vector reranking over BGE-Dense candidates.
    if args.include_multivector:
        mv_extra = (
            f" --model_name {args.mv_model_name}"
            f" --max_seq_length {args.mv_max_seq_length}"
            f" --doc_batch_size {args.mv_doc_batch_size}"
            f" --query_batch_size {args.mv_query_batch_size}"
            f" --rerank_batch_size {args.mv_rerank_batch_size}"
            f"{device_arg}"
        )

        for qlang in query_langs:
            # MV base reranking for triples/text/concat.
            for dlang in triple_doc_langs:
                cand = path("bge_m3_dense", "triples", qlang, dlang)
                out = path("bge_m3_multivector", "triples", qlang, dlang)
                print(
                    f"python run_multivector_bge_rerank.py --data_dir {args.data_dir} "
                    f"--query_lang {qlang} --doc_lang {dlang} --representation triples --split {args.split} "
                    f"--candidate_run {cand} --candidate_top_n {args.top_k_base} "
                    f"--top_k {args.top_k_final} --output {out}{mv_extra}{profile_arg}"
                )
            for dlang in required_text_doc_langs:
                cand = path("bge_m3_dense", "text", qlang, dlang)
                out = path("bge_m3_multivector", "text", qlang, dlang)
                print(
                    f"python run_multivector_bge_rerank.py --data_dir {args.data_dir} "
                    f"--query_lang {qlang} --doc_lang {dlang} --representation text --split {args.split} "
                    f"--candidate_run {cand} --candidate_top_n {args.top_k_base} "
                    f"--top_k {args.top_k_final} --output {out}{mv_extra}{profile_arg}"
                )
            for dlang in concat_doc_langs:
                cand = path("bge_m3_dense", "concat", qlang, dlang)
                out = path("bge_m3_multivector", "concat", qlang, dlang)
                print(
                    f"python run_multivector_bge_rerank.py --data_dir {args.data_dir} "
                    f"--query_lang {qlang} --doc_lang {dlang} --representation concat --split {args.split} "
                    f"--candidate_run {cand} --candidate_top_n {args.top_k_base} "
                    f"--top_k {args.top_k_final} --output {out}{mv_extra}{profile_arg}"
                )

            # Hybrid cascade and MV late fusion.
            for dlang in hybrid_doc_langs:
                tlang = text_lang_for_hybrid(dlang)
                tri = path("bge_m3_dense", "triples", qlang, dlang)
                txt = path("bge_m3_dense", "text", qlang, tlang)
                union = path("bge_m3_dense", "hybrid_union_candidates", qlang, dlang)
                cascade = path("bge_m3_multivector", "hybrid_cascade", qlang, dlang)
                print(
                    f"python run_hybrid_fusion.py --fusion union --run_triples {tri} --run_text {txt} "
                    f"--query_lang {qlang} --doc_lang {dlang} --candidate_top_n {args.top_k_base} "
                    f"--top_k {args.top_k_base} --output {union}{profile_arg}"
                )
                print(
                    f"python run_multivector_bge_rerank.py --data_dir {args.data_dir} "
                    f"--query_lang {qlang} --doc_lang {dlang} --representation concat --split {args.split} "
                    f"--candidate_run {union} --candidate_top_n {args.top_k_base} "
                    f"--top_k {args.top_k_final} --output {cascade}{mv_extra}{profile_arg}"
                )

                mv_tri = path("bge_m3_multivector", "triples", qlang, dlang)
                mv_txt = path("bge_m3_multivector", "text", qlang, tlang)
                mv_rrf = path("bge_m3_multivector", "hybrid_rrf", qlang, dlang)
                mv_score = path("bge_m3_multivector", "hybrid_score", qlang, dlang)
                print(
                    f"python run_hybrid_fusion.py --fusion rrf --run_triples {mv_tri} --run_text {mv_txt} "
                    f"--query_lang {qlang} --doc_lang {dlang} --top_k {args.top_k_final} --output {mv_rrf}{profile_arg}"
                )
                print(
                    f"python run_hybrid_fusion.py --fusion score --run_triples {mv_tri} --run_text {mv_txt} "
                    f"--query_lang {qlang} --doc_lang {dlang} --top_k {args.top_k_final} --output {mv_score}{profile_arg}"
                )

    ks = " ".join(map(str, args.ks))
    print(
        f"python evaluate_runs.py --data_dir {args.data_dir} "
        f"--runs '{R}/*.{args.ext}' --out_dir {args.eval_dir} --ks {ks}"
    )
    if args.profile:
        print(
            f"python summarize_resource_profiles.py --runs_dir {R} "
            f"--out_csv {Path(args.eval_dir) / 'resource_summary.csv'}"
        )


if __name__ == "__main__":
    main()
