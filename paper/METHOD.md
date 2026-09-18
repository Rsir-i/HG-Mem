# Method Details

This document is a concise, self-contained description of HG-Mem. It mirrors
the **Method** section of the paper. If anything here disagrees with the
paper, the paper takes precedence.

## 1. Problem Statement

Given a long-term dialogue history `H = {S₁, S₂, …, Sₙ}`, where each session
`Sᵢ = {tᵢ,₁, tᵢ,₂, …, tᵢ,ⱼ}` is a sequence of dialogue turns, and a current
query `q`, the goal of long-term conversational memory retrieval is to
return a small set `ε_q = R(q, H)` of **top-*k* evidence turns** that are
relevant to `q`. The final retrieval unit is always an *original turn* —
session summaries and topic-cluster representations are compressed and are
therefore not returned directly.

## 2. Hierarchical Memory Graph

HG-Mem represents the dialogue history as a hierarchical graph

```
G = (V, E),   V = V_S ∪ V_C ∪ V_T
```

where

- `V_S` is the set of **session nodes**;
- `V_C` is the set of **topic-cluster nodes**;
- `V_T` is the set of **turn nodes**.

### 2.1 Session and turn nodes (Eqs. in §3.2.2 of the paper)

For each session `Sᵢ`, an LLM generates a summary `dᵢ` and a semantic
encoder `ϕ(·)` produces the session vector `eᵢ^S = ϕ(dᵢ)`. The session
node is

```
vᵢ^S = (dᵢ, eᵢ^S).
```

For each turn `tᵢ,ⱼ`, the original text is retained as evidence and
embedded as `eᵢ,ⱼ^T = ϕ(tᵢ,ⱼ)`. The turn node is

```
vᵢ,ⱼ^T = (tᵢ,ⱼ, eᵢ,ⱼ^T).
```

### 2.2 Topic-cluster nodes (Eqs. 3-4)

For every turn `tᵢ,ⱼ`, an LLM produces a topic description `zᵢ,ⱼ`.
Clustering is performed **independently per session** to avoid topic
confusion across sessions. The MiniLM encoder embeds each topic, and
**hierarchical agglomerative clustering with average linkage and cosine
distance** merges turns whose topics are close (distance threshold =
0.55; isolated clusters are allowed).

The r-th topic-cluster of session `Sᵢ` is

```
Cᵢ,ᵣ = { zᵢ,ⱼ | πᵢ,ⱼ = r },
```

and its node is

```
vᵢ,ᵣ^C = (eᵢ,ᵣ^C),   where   eᵢ,ᵣ^C = (1 / |Cᵢ,ᵣ|) · Σ ϕ(zᵢ,ⱼ).
```

The cluster text is set to the longest topic phrase inside the cluster.

### 2.3 Edges (Eqs. 5-7)

Two entailment edges connect high-level nodes to original turns:

- Session → Turn:

```
E_{S→T} = { (vᵢ^S, vᵢ,ⱼ^T) | tᵢ,ⱼ ∈ Sᵢ }.
```

- Topic-Cluster → Turn:

```
E_{C→T} = { (vᵢ,ᵣ^C, vᵢ,ⱼ^T) | tᵢ,ⱼ ∈ Cᵢ,ᵣ }.
```

## 3. Three-Path Candidate Recall (§3.3.1)

For a query `q`, HG-Mem recalls candidate turns through three paths and
merges the results into a unified candidate set `C_q = C_T ∪ C_S ∪ C_C`.

### 3.1 BM25 path (Eq. 8)

```
s_BM25(q, vᵢ,ⱼ^T) = BM25(q, tᵢ,ⱼ)
```

Top-`N_T` (= 100) turn nodes with the highest `s_BM25` form `C_T`.

### 3.2 Session path (Eq. 9)

```
s_session(q, vᵢ^S) = cos(ϕ(q), eᵢ^S)
```

Top-`N_S` (= 3) sessions with the highest `s_session` are expanded via
`E_{S→T}` to all of their constituent turns. This set is `C_S`.

### 3.3 Topic-cluster path (Eq. 10)

```
s_topic(q, vᵢ,ᵣ^C) = cos(ϕ(q), eᵢ,ᵣ^C)
```

Top-`N_C` (= 5) topic-clusters with the highest `s_topic` are expanded via
`E_{C→T}`. This set is `C_C`. Unlike the session path, the topic path
expands only a *local topical region*, reducing irrelevant history.

## 4. Multi-Signal Ranking (§3.3.2)

For every candidate turn `tᵢ,ⱼ ∈ C_q`, HG-Mem computes three ranking
signals.

### 4.1 BM25 lexical signal (Eq. 11)

```
s₁(q, vᵢ,ⱼ^T) = BM25(q, tᵢ,ⱼ).
```

### 4.2 Turn-level semantic signal (Eq. 12)

```
s₂(q, vᵢ,ⱼ^T) = cos(ϕ(q), eᵢ,ⱼ^T).
```

### 4.3 Topic-aware semantic signal (Eq. 13)

```
s₃(q, vᵢ,ⱼ^T) = cos(ϕ(q), eᵢ,πᵢ,ⱼ^C).
```

If a turn has no valid topic-cluster representation, `s₃` is set to 0.

### 4.4 Min-max normalisation (Eq. 14)

Because the three signals have different scales, each signal is min-max
normalised within `C_q`:

```
~s_m(q, tᵢ,ⱼ) = (s_m(q, tᵢ,ⱼ) − min_{t'∈C_q} s_m) /
                (max_{t'∈C_q} s_m − min_{t'∈C_q} s_m + ε).
```

### 4.5 Weighted fusion (Eq. 15)

The final score is a weighted sum of the three normalised signals:

```
Score(q, tᵢ,ⱼ) = α · ~s₁ + β · ~s₂ + γ · ~s₃,
```

with `α + β + γ = 1`. The top-*k* turns by `Score` are returned as
`ε_q`.

In the paper, the optimal weights are obtained by grid-search:

- LoCoMo: (α, β, γ) = **(0.4, 0.4, 0.2)**.
- LongMemEval: (α, β, γ) = **(0.5, 0.4, 0.1)**.

## 5. End-to-End QA

For end-to-end question answering, the top-*k* retrieved turns are
concatenated into a memory context and fed to Qwen2.5-7B-Instruct. The
prompt asks the model to "answer as concisely as possible — just give the
direct answer, no explanation". `k = 10` for both datasets. The
F1-score is computed as the token-level F1 between the generated answer
and the reference answer.

## 6. Implementation Notes

- Semantic encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
  Maximum input length 512, batch size 32, attention-mask-weighted mean
  pooling.
- BM25: a from-scratch implementation that uses stop-word removal and
  (for LongMemEval) Porter stemming; `k₁ = 1.5`, `b = 0.75`.
- Topic clustering: `sklearn.cluster.AgglomerativeClustering` with
  `metric='precomputed'`, `linkage='average'`,
  `distance_threshold=0.55`. Cosine distance matrix is converted from
  the MiniLM topic embeddings.
- Topic generation prompt: a single prompt per session asks Qwen to
  emit a 5–10-word topic phrase for every turn in batches of ≤ 20 turns
  (to avoid the model "skipping" long sessions).
- Session-summary generation prompt: Qwen is asked to write 3–5
  sentences that preserve all proper entities, numbers, and dates.
- All caches (`*_cache*.json`, `*_result*.json`) are regenerated on demand
  and are not committed to the repository.

## 7. Complexity Analysis (§3.4)

HG-Mem concentrates its computational overhead in the **offline
construction** stage (session-summary generation, topic generation, and
agglomerative clustering per session). Online inference then reduces to:

1. A BM25 pass over the inverted index.
2. A single `cos(ϕ(q), ·)` computation against `|V_S| + |V_C|` vectors.
3. Expansion of the top sessions / topic-clusters to candidate turns.
4. Linear weighted fusion of three min-max-normalised signals.

Because steps 1-4 are all linear (or sub-linear) in the candidate-pool
size, HG-Mem's online retrieval overhead is small — about **0.32 s per
query on LoCoMo and 0.43 s per query on LongMemEval** on a single
RTX 4090. This makes the method well-suited to scenarios where the same
memory store is queried repeatedly, so that the one-time offline
construction cost can be amortised.