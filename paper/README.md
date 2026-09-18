# Paper Materials

This directory contains the manuscript and figures associated with the paper:

> **HG-Mem: Hierarchical Graph and Multi-Path Retrieval based Long-Term
> Memory Management for Conversational Agents**

| File         | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| `HG-Mem.pdf` | Manuscript of the paper (pre-print PDF).                          |
| `method.png` | Figure 1 — the overall framework of HG-Mem (hierarchical graph + three-path retrieval + multi-signal ranking). |

## Figure 1 — `method.png`

![HG-Mem framework](method.png)

The figure illustrates the two-stage pipeline of HG-Mem:

1. **Hierarchical Memory Construction** — session, topic-cluster, and turn
   nodes are organized into a hierarchical memory graph, with two types of
   entailment edges linking high-level semantic representations to the
   original turn-level evidence.

2. **Hierarchical-Graph-based Retrieval** — given a user query, candidate
   turns are recalled through three complementary paths (BM25, session
   path, and topic-cluster path), then re-ranked using three signals
   (BM25 lexical score, turn-level semantic score, topic-aware semantic
   score) to produce the final top-*k* evidence turns.

If you need to reference the figure in your own work, please cite the paper
listed in the top-level [`README.md`](../README.md#citation).