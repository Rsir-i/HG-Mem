# Reproducing the Experiments

This directory contains the two experiment scripts used to produce every
table and figure reported in the paper.

| Script                                | Dataset       |
| ------------------------------------- | ------------- |
| `graph_vs_baselines_LoCoMo.py`        | LoCoMo        |
| `graph_vs_baselines_LongMemEval.py`   | LongMemEval (`longmemeval_s_cleaned`) |

The three baseline memory systems that HG-Mem is compared against live in their own
sub-directories. Each is a trimmed copy of the corresponding official release and is run
with its own harness:

| Baseline | Folder | README |
| -------- | ------ | ------ |
| A-Mem (Agentic Memory) | `A-MEM/` | [`A-MEM/README.md`](A-MEM/README.md) |
| MemGAS | `MemGAS/` | [`MemGAS/README.md`](MemGAS/README.md) |
| SeCom | `Secom/` | [`Secom/README.md`](Secom/README.md) |

They are evaluated on the same splits under `../dataset/` (download them once with the
links in [`../dataset/README.md`](../dataset/README.md)), but each folder documents its
own dependencies and command-line flags.

Both scripts share the same overall structure: an **offline construction
phase** that builds the hierarchical memory graph and pre-computes the
embeddings / BM25 indices, followed by a **grid-search phase** that
searches over the three ranking weights, a **retrieval-evaluation phase**
that computes R@k and MRR, an optional **end-to-end phase** that uses
Qwen2.5-7B-Instruct to generate answers and computes F1 / EM, and a final
**report phase** that prints the results and saves them to disk.

The scripts are **self-contained** — they only depend on `numpy`, `torch`,
`transformers`, `scikit-learn`, and `nltk` (see [`requirements.txt`](requirements.txt)).

## Common Command-Line Flags

| Flag                 | Default                                         | Description                                                  |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `--data_path`        | dataset-specific (see below)                    | Path to the input dataset.                                   |
| `--qwen_path`        | `None`                                          | Path to a local Qwen2.5-7B-Instruct checkpoint. **Required** for the first run (to generate per-turn topics and, for LongMemEval, session summaries). |
| `--weight_search`    | *off*                                           | Grid-search the three ranking weights over [0, 1] with the constraint α+β+γ=1. |
| `--grid_step`        | `0.10`                                          | Step size for the weight grid.                               |
| `--sem_session_k`    | `3`                                             | Number of sessions retrieved by the session path.           |
| `--sem_topic_k`      | `5`                                             | Number of topic-clusters retrieved by the topic path.        |
| `--sem_bm25_n`       | `100`                                           | Number of turns retrieved by the BM25 path.                  |
| `--end_to_end`       | *off*                                           | Also run end-to-end answer generation + F1/EM evaluation. **Requires `--qwen_path`.** |
| `--max_retrieved`    | `10`                                            | Number of top-ranked turns fed to the answer-generation model in the E2E phase. |

Dataset-specific flags are documented in the corresponding scripts.

## LoCoMo — `graph_vs_baselines_LoCoMo.py`

### Default arguments

```bash
--data_path       ../dataset/locomo10.json
--turn_topic_cache turn_topics_cache.json
```

### Stage 1 — generate the turn-topic cache (one-time)

```bash
python graph_vs_baselines_LoCoMo.py \
    --data_path ../dataset/locomo10.json \
    --qwen_path /path/to/Qwen2.5-7B-Instruct \
    --weight_search
```

The first run loads Qwen2.5-7B-Instruct, asks it to produce a 5–10-word
topic phrase for every turn in batches of ≤ 20, and saves the result to
`turn_topics_cache.json`. Session summaries are **not** generated here —
the script uses the session summaries shipped with LoCoMo directly.

### Stage 2 — reproduce the retrieval / weight-search results

```bash
python graph_vs_baselines_LoCoMo.py \
    --data_path ../dataset/locomo10.json \
    --weight_search
```

After the cache exists, Qwen is no longer needed. The script will:

1. Build the hierarchical memory graph for each conversation.
2. Grid-search the ranking weights `(α, β, γ)` over [0, 1] with step 0.1
   and α+β+γ=1 (66 valid combinations in total). Recall@10 is used to
   select the best combination.
3. Evaluate every method on Recall@1 / 3 / 5 / 10 and MRR@10.
4. Print per-category Recall@5 results and the average retrieval time
   per query.
5. Save the full result to `graph_turn_topic_recall_result.json`.

### Stage 3 — add end-to-end QA + F1/EM

```bash
python graph_vs_baselines_LoCoMo.py \
    --data_path ../dataset/locomo10.json \
    --qwen_path /path/to/Qwen2.5-7B-Instruct \
    --weight_search \
    --end_to_end
```

The top-10 retrieved turns of each method are concatenated and fed into
Qwen2.5-7B-Instruct as the memory context; the F1-score and exact-match
metrics are computed against the reference answers in the `qa` field.

## LongMemEval — `graph_vs_baselines_LongMemEval.py`

### Default arguments

```bash
--data_path         ../dataset/longmemeval_s_cleaned.json
--summary_cache     longmemeval_summaries_v3.json
--turn_topic_cache  turn_topics_cache_v3.json
```

### Stage 1 — generate the caches (one-time)

```bash
python graph_vs_baselines_LongMemEval.py \
    --data_path ../dataset/longmemeval_s_cleaned.json \
    --qwen_path /path/to/Qwen2.5-7B-Instruct \
    --weight_search
```

For LongMemEval, HG-Mem generates **two** caches:

- `longmemeval_summaries_v3.json` — a session summary for every haystack
  session (3–5 sentences; the prompt instructs Qwen to preserve all
  specific entities).
- `turn_topics_cache_v3.json` — a topic phrase for every turn, keyed by
  `f"{session_id}||{session_idx}_{msg_idx}"`.

### Stage 2 — retrieval / weight-search

```bash
python graph_vs_baselines_LongMemEval.py \
    --data_path ../dataset/longmemeval_s_cleaned.json \
    --weight_search
```

The script will grid-search the three weights, evaluate every method, and
save the full result to `graph_hybrid_recall_v3_update_result.json`.

### Stage 3 — end-to-end QA

```bash
python graph_vs_baselines_LongMemEval.py \
    --data_path ../dataset/longmemeval_s_cleaned.json \
    --qwen_path /path/to/Qwen2.5-7B-Instruct \
    --weight_search \
    --end_to_end
```

The top-10 retrieved turns of each method are concatenated and fed into
Qwen2.5-7B-Instruct as the memory context for end-to-end F1 / EM
evaluation.

## Hardware

All experiments in the paper were conducted on a server with a 14-core
AMD EPYC 7453 CPU and a single NVIDIA RTX 4090 GPU (24 GB VRAM) for
PyTorch-based training and inference.

## What the Scripts Print / Save

Both scripts print, at the end of every run, a summary that includes:

- Overall Recall@1, R@3, R@5, R@10, and MRR for every method (HG-Mem and
  every baseline / ablation).
- End-to-end F1 / EM and generation / retrieval timing if `--end_to_end`
  was specified.
- Per-category Recall@5 for the LoCoMo `category` field or the
  LongMemEval `question_type` field.
- The complete weight grid sorted by R@10 when `--weight_search` is set.

The full structured result is also written to a JSON file in the working
directory:

- `graph_turn_topic_recall_result.json` for LoCoMo.
- `graph_hybrid_recall_v3_update_result.json` for LongMemEval.

## Files Generated at Runtime


| File                                       | Purpose                                                |
| ------------------------------------------ | ------------------------------------------------------ |
| `turn_topics_cache.json`                   | Cached Qwen-generated turn topics for LoCoMo.          |
| `longmemeval_summaries_v3.json`            | Cached Qwen-generated session summaries for LongMemEval.|
| `turn_topics_cache_v3.json`                | Cached Qwen-generated turn topics for LongMemEval.     |
| `graph_turn_topic_recall_result.json`      | Final result file for the LoCoMo experiment.           |
| `graph_hybrid_recall_v3_update_result.json`| Final result file for the LongMemEval experiment.      |