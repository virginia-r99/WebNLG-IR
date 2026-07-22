# WebNLG-IR


**WebNLG-IR** is a multilingual information retrieval benchmark and experimental toolkit for studying how knowledge-graph facts should be represented for textual retrieval: as textual triples, as natural-language verbalisation, or as a combination of both.

The benchmark is derived from WebNLG and its Spanish adaptation. It provides English, Spanish, and Catalan questions as queries, together with English, Spanish, and mixed-language knowledge-graph document conditions. The experiments compare lexical, dense, learned-sparse, multi-vector, and hybrid retrieval across different document-complexity levels.

## Main finding

Across retrieval methods, query languages, document-language conditions, question types, and tripleset-complexity levels, **concatenating triples and verbalised text in the same retrievable document is the strongest overall representation**. Triples generally outperform verbalised text alone, while early triple–text fusion provides complementary structured and natural-language evidence more effectively than late rank or score fusion.

## Why WebNLG-IR?

Knowledge graphs are commonly accessed through formal query languages such as SPARQL. This can be difficult for non-expert users, requires knowledge of the graph schema, and may be impractical when a maintained endpoint or stable formalisation is unavailable. WebNLG-IR studies an alternative access setting in which natural-language queries retrieve KG-derived documents using standard information retrieval methods.

The benchmark is designed to answer the following question:

> Should KG facts be retrieved as compact triples, as natural-language lexicalisations, or through a hybrid triple–text representation?

## Benchmark summary

| Statistic | Value |
|---|---:|
| Prepared queries | 35,616 |
| Canonical documents | 16,631 |
| Relevance judgements | 503,985 |
| Query languages | English, Spanish, Catalan |
| Document conditions | English, Spanish, mixed English–Spanish triples |
| Extractive queries | 17,451 |
| Negative yes/no queries | 9,093 |
| Positive yes/no queries | 9,072 |
| Documents with at least 2 triples | 12,664 |
| Documents with at least 3 triples | 9,530 |
| Documents with at least 4 triples | 6,125 |

Answers are retained in the source QA data for traceability and possible future QA experiments, but **they are not used for retrieval or evaluation**.

## Dataset construction

WebNLG-IR is built in four stages:

1. **Question generation.** English extractive and yes/no questions are generated from single-triple WebNLG lexicalisations.
2. **Validation and translation.** The questions are automatically validated and translated into Spanish and Catalan while preserving entities, values, units, and titles.
3. **Document construction.** WebNLG entries are represented as English triples, Spanish triples, aligned lexicalisations, and mixed-language triplesets.
4. **Relevance projection.** A query is relevant to every canonical WebNLG document whose tripleset contains the query's source triple.

This projection creates controlled single-hop information needs while allowing retrieval to be evaluated over larger multi-triple documents.

### Mixed-language triples

For the `mix` condition, aligned English and Spanish triples are combined within the same tripleset. For each aligned position, the English or Spanish version of the same fact is selected while preserving:

- the original entry;
- the original fact inventory;
- the aligned triple position;
- the original tripleset order.

The language allocation is deterministic given the configured random seed. For mixed-language hybrid and concatenated representations, the paired verbalisation is English.

## Query and document conditions

Queries are available in:

- `en`: English (`question`)
- `es`: Spanish (`question_es`)
- `ca`: Catalan (`question_ca`)

Document conditions are:

| Condition | Triple representation | Text representation |
|---|---|---|
| `en` | English/canonical triples | English lexicalisation |
| `es` | Spanish triples | Spanish lexicalisation |
| `mix` | Mixed English–Spanish triples | English lexicalisation |

The `mix` text field is equivalent to the English text by design. It is used to construct `concat_mix` and mixed hybrid runs, but it should not be interpreted as a third independent text-only language condition.

The full matrix therefore supports monolingual and cross-lingual directions such as:

- English query → English, Spanish, or mixed documents
- Spanish query → English, Spanish, or mixed documents
- Catalan query → English, Spanish, or mixed documents

## Document representations

| Representation | Description |
|---|---|
| `triples` | The complete tripleset serialised as textual subject–predicate–object statements. |
| `text` | One natural-language lexicalisation of the WebNLG entry. |
| `concat` | Early fusion: `[TRIPLES] <triples> [TEXT] <lexicalisation>`. |
| `hybrid_rrf` | Weighted reciprocal-rank fusion of separate triple and text rankings. |
| `hybrid_score` | Weighted fusion of separately normalised triple and text scores. |
| `hybrid_cascade` | Optional candidate union followed by BGE-M3 multi-vector reranking. |

The main comparison is between triples, text, concatenation, RRF, and score fusion. The cascade implementation is included as an additional experimental option.

## Retrieval methods

The repository includes:

| Script | Retrieval paradigm |
|---|---|
| `run_bm25.py` | Lexical BM25 retrieval |
| `run_dense_bge.py` | BGE-M3 dense bi-encoder retrieval |
| `run_sparse_milco.py` | MILCO learned-sparse multilingual retrieval |
| `run_multivector_bge_rerank.py` | BGE-M3 multi-vector late-interaction reranking |
| `run_hybrid_fusion.py` | RRF, score fusion, and candidate union |

Neural scripts support GPU execution, batched encoding, and configurable query-scoring chunks.

## Repository structure

```text
WebNLG-IR/
├── code/
│   ├── dataset_dev/
│   │   ├── document_extraction_mapping_lang3.py
│   │   ├── question generation/validation scripts and notebooks
│   │   └── intermediate development resources
│   └── evaluation/
│       ├── prepare_webnlg_ir.py
│       ├── make_matrix_commands.py
│       ├── run_bm25.py
│       ├── run_dense_bge.py
│       ├── run_sparse_milco.py
│       ├── run_multivector_bge_rerank.py
│       ├── run_hybrid_fusion.py
│       ├── evaluate_runs.py
│       ├── resource_profiler.py
│       ├── summarize_resource_profiles.py
│       └── evaluation_notebook.ipynb
├── dataset/
│   ├── dataset_lang3/
│   ├── prepared_min2_mixed_normtext/
│   ├── prepared_min3_mixed_normtext/
│   └── prepared_min4_mixed_normtext/
├── results/
│   ├── eval_min2_mixed_normtext/
│   ├── eval_min3_mixed_normtext/
│   └── eval_min4_mixed_normtext/
├── LICENSE
└── README.md
```

The main reproducible evaluation pipeline is under `code/evaluation/`. The `code/dataset_dev/` directory contains the dataset-generation pipeline and intermediate development scripts.

## Installation

Python 3.12 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install \
  pandas numpy scipy tqdm pyarrow psutil \
  torch sentence-transformers transformers FlagEmbedding \
  matplotlib jupyter
```

Notes:

- BM25 runs on CPU.
- CUDA is strongly recommended for BGE-M3, MILCO, and multi-vector reranking.
- Model weights are downloaded on first use from their respective model repositories.
- MILCO is loaded with `trust_remote_code=True`; inspect remote model code according to your security requirements.

## Using the included prepared data

Prepared collections are provided for the three document-complexity settings used in the experiments:

```text
dataset/prepared_min2_mixed_normtext
dataset/prepared_min3_mixed_normtext
dataset/prepared_min4_mixed_normtext
```

Each prepared directory contains:

```text
docs.parquet       # one row per canonical WebNLG document
queries.parquet    # multilingual queries and metadata
qrels.parquet      # binary relevance judgements
manifest.json      # preparation parameters and collection statistics
```

The document table contains language-specific fields such as:

```text
triples_en, triples_es, triples_mix
text_en, text_es, text_mix
concat_en, concat_es, concat_mix
```

## Rebuilding a prepared collection

From `code/evaluation/`:

```bash
python prepare_webnlg_ir.py \
  --qa_csv '../../dataset/dataset_lang3/webnlg_qa_selected_es_v4_validated_lang3_min1.csv' \
  --triplesets_csv '../../dataset/dataset_lang3/ir_triplesets_lang3_min1.csv' \
  --texts_csv '../../dataset/dataset_lang3/ir_texts_lang3_min1.csv' \
  --out_dir '../../dataset/prepared_min2_mixed_normtext' \
  --min_triples 2 \
  --doc_langs en es mix \
  --triple_variant normalized \
  --text_variant normalized
```

Repeat with `--min_triples 3` and `--min_triples 4` for the other complexity levels.

### Meaning of `--min_triples`

By default, `--min_triples K` filters **query eligibility**: a query is retained only when it has at least one relevant document containing at least `K` triples. The complete document pool is retained to preserve non-relevant distractors and task difficulty.

All relevant documents in the pool are included in the qrels for retained queries. To physically remove documents below the threshold as a separate ablation, also pass:

```bash
--filter_docs_by_min_triples
```

### Representation preprocessing

Triple fields support:

```text
--triple_variant raw|normalized|both
```

Text fields support:

```text
--text_variant raw|normalized|both
```

Text normalisation is deliberately light: Unicode normalisation, lowercasing, underscore replacement, punctuation cleanup, and whitespace collapse. The pipeline does **not** perform stopword removal, stemming, lemmatisation, or translation.

## Running the full experimental matrix

From `code/evaluation/`, generate a shell script:

```bash
python make_matrix_commands.py \
  --data_dir '../../dataset/prepared_min2_mixed_normtext' \
  --results_dir '../../runs/min2_mixed_normtext' \
  --eval_dir '../../results/eval_min2_mixed_normtext' \
  --split ALL \
  --query_langs en es ca \
  --doc_langs en es mix \
  --top_k_base 1000 \
  --top_k_final 100 \
  --ks 5 10 100 \
  --device cuda \
  --doc_batch_size 128 \
  --query_batch_size 128 \
  --query_chunk_size 1024 \
  --mv_doc_batch_size 64 \
  --mv_query_batch_size 32 \
  --mv_rerank_batch_size 64 \
  --profile \
  > run_matrix_min2_mixed_normtext.sh

bash run_matrix_min2_mixed_normtext.sh
```

Adjust batch sizes if CUDA runs out of memory. Use `--no_milco` or `--no_multivector` to skip the most resource-intensive components.

### Running a single condition

Example: Catalan queries against the mixed triples-plus-English-text concatenation using BM25.

```bash
python run_bm25.py \
  --data_dir '../../dataset/prepared_min2_mixed_normtext' \
  --query_lang ca \
  --doc_lang mix \
  --representation concat \
  --split ALL \
  --top_k 100 \
  --output '../../runs/min2_mixed_normtext/bm25__concat__qca_dmix.parquet'
```

## Hybrid construction

Hybrids combine aligned representations from the same document condition:

```text
hybrid_en  = triples_en  + text_en
hybrid_es  = triples_es  + text_es
hybrid_mix = triples_mix + text_en
```

The code does not create an arbitrary multilingual document pool. Query language and document condition are varied independently, enabling controlled monolingual and cross-lingual evaluation.

## Evaluation

`evaluate_runs.py` evaluates binary qrels at the canonical document level. It uses **all relevant documents** for each query when computing denominators.

Supported metrics include:

- Average Precision (`AP`)
- Precision at `k`
- Recall at `k`
- nDCG at `k`
- MRR at `k`

Example:

```bash
python evaluate_runs.py \
  --data_dir '../../dataset/prepared_min2_mixed_normtext' \
  --runs '../../runs/min2_mixed_normtext/*.parquet' \
  --out_dir '../../results/eval_min2_mixed_normtext' \
  --ks 5 10 100
```

Evaluation outputs include:

```text
per_query_metrics.parquet / .csv
summary_metrics.parquet / .csv
resource_summary.csv
resource_summary_by_method_representation.csv
```

## Resource profiling

Pass `--profile` to the matrix generator or an individual retrieval script to record:

- total wall-clock time;
- indexing or document-encoding time;
- query-encoding and scoring time;
- documents and queries processed per second;
- peak process RSS memory;
- peak CUDA allocated and reserved memory;
- GPU and device metadata.

Profiles are stored beside run files as `*.profile.json` and aggregated with:

```bash
python summarize_resource_profiles.py \
  --runs_dir '../../runs/min2_mixed_normtext' \
  --out_csv '../../results/eval_min2_mixed_normtext/resource_summary.csv'
```

## Results

The main empirical pattern is consistent across the evaluated settings:

1. **Verbalised text alone is generally the weakest representation.**
2. **Triples outperform text alone**, suggesting that compact subject–predicate–object structure remains valuable for retrieval.
3. **Concatenation performs best overall**, across retrieval methods and the `≥2`, `≥3`, and `≥4` complexity settings.
4. **Late fusion improves over text alone but generally remains below concatenation.** Score fusion is usually slightly stronger than RRF.
5. **The concatenation advantage persists across English, Spanish, and Catalan queries**, and across English, Spanish, and mixed-language document conditions.
6. **The same pattern appears for extractive, positive yes/no, and negative yes/no questions.**

These findings indicate that data-to-text verbalisation is most useful as complementary evidence rather than as a replacement for triples. Triples expose explicit relational structure, while lexicalisations add natural-language context and alternative lexical cues. Early fusion lets a retriever match both forms jointly.

## Result analysis notebook

Use:

```text
code/evaluation/evaluation_notebook.ipynb
```

The notebook supports comparisons by:

- representation;
- retrieval method;
- query language;
- document condition;
- question type;
- tripleset complexity (`min2`, `min3`, `min4`);
- retrieval effectiveness versus runtime and memory.

## Reproducibility notes

- All original WebNLG splits are used by default (`--split ALL`).
- Queries are filtered to ensure relevance coverage at the selected complexity threshold.
- The document pool remains unchanged unless explicitly filtered.
- Mixed triples are generated deterministically from aligned English and Spanish triples using a fixed seed.
- Qrels retain every relevant canonical document, not only one supporting document.
- Run metadata records method, representation, query language, document condition, split, and question type.

## Limitations

WebNLG-IR is an automatically constructed benchmark intended primarily for controlled comparison of document representations. Current limitations include:

- questions, answers, and projected relevance judgements have not yet undergone exhaustive human validation;
- the mixed-language condition assumes positional alignment between English and Spanish triples;
- the benchmark is derived from WebNLG and therefore does not cover all KG domains or schema characteristics;
- one selected lexicalisation represents each entry and language in the retrieval collection;
- answers are not exhaustively validated because answering is outside the present IR task.

## Licence

Repository code is released under the [Apache License 2.0](LICENSE). Derived datasets may also be subject to the licences and attribution requirements of the original WebNLG resources from which they were constructed.


## Acknowledgements

This work builds on WebNLG and its Spanish adaptation. AI-assisted tools were used to help implement code and refine language. All suggestions were reviewed and finalised by the authors.
