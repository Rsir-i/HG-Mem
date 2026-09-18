import numpy as np


def dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.


def ndcg(rankings, correct_docs, corpus_ids, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg


def mrr(rankings, correct_docs, corpus_ids, k=None):
    """Mean Reciprocal Rank. If k is given, only consider top-k results."""
    rankings = rankings[:k] if k is not None else rankings
    for rank, idx in enumerate(rankings, start=1):
        if corpus_ids[idx] in correct_docs:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    """Standard retrieval evaluation: recall = |retrieved ∩ relevant| / |relevant|"""
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    num_relevant = len(correct_docs)
    num_hit = len(recalled_docs & set(correct_docs))
    recall = num_hit / num_relevant if num_relevant > 0 else 0.0
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    mrr_score = mrr(rankings, correct_docs, corpus_ids, k=k)
    return recall, ndcg_score, mrr_score


def evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=10):
    # convert turn-level labels/results into session-level and then evaluate
    def strip_turn_id(docid):
        return '_'.join(docid.split('_')[:-1])
    correct_docs = list(set([strip_turn_id(x) for x in correct_docs]))

    # revise k to handle document-level retrieval
    corpus_ids = [strip_turn_id(x) for x in corpus_ids]
    effective_k = k
    unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    while effective_k <= len(corpus_ids) and len(unique_docids) < k:
        effective_k += 1
        unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])

    return evaluate_retrieval(rankings, correct_docs, corpus_ids, k=effective_k)

