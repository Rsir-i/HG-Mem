import re
import string
import numpy as np
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from sentence_transformers.util import pytorch_cos_sim

# Download required NLTK data
for resource in ['punkt_tab', 'punkt', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Initialize SentenceTransformer model (this will be reused)
_sbert_model_name = 'all-MiniLM-L6-v2'
sentence_model = None

def set_sbert_model_path(model_name_or_path: str):
    """Set the SentenceTransformer model path before first use.
    Call this before calculate_metrics() to use a local model.
    """
    global _sbert_model_name, sentence_model
    _sbert_model_name = model_name_or_path
    sentence_model = None  # force reload on next use

def _get_sbert_model():
    global sentence_model
    if sentence_model is None:
        try:
            sentence_model = SentenceTransformer(_sbert_model_name)
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model: {e}")
            sentence_model = None
    return sentence_model

def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores

def calculate_bert_scores(prediction: str, reference: str,
                          model_type: str = "all-MiniLM-L6-v2") -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity.
    
    Uses a local embedding model to avoid downloading roberta-large (the default).
    Falls back to zeros if the model is unavailable.
    """
    try:
        P, R, F1 = bert_score([prediction], [reference], model_type=model_type,
                              lang='en', verbose=False)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore (model={model_type}): {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    model = _get_sbert_model()
    if model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = model.encode([prediction], convert_to_tensor=True)
        embedding2 = model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0
        }
    
    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate all scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference)
    
    # Combine all metrics (BERTScore disabled)
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}
    
    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))
    
    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)
    
    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results


# ---------------------------------------------------------------------------
# Retrieval evaluation metrics: R@1, R@3, R@5, R@10, MRR@10
# ---------------------------------------------------------------------------

def calculate_retrieval_metrics(
    retrieved_indices: List[int],
    gt_indices: List[int],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Calculate retrieval metrics for a single query.

    Args:
        retrieved_indices: ordered list of retrieved memory indices (top-K)
        gt_indices: list of ground-truth memory indices (evidence)
        k_values: list of K values for R@K (default: [1, 3, 5, 10])

    Returns:
        dict with keys like "R@1", "R@3", "R@5", "R@10", "MRR@10"
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if not gt_indices:
        return {f"R@{k}": 0.0 for k in k_values} | {"MRR@10": 0.0}

    gt_set = set(gt_indices)
    metrics = {}

    # Recall@K
    for k in k_values:
        top_k = set(retrieved_indices[:k])
        metrics[f"R@{k}"] = 1.0 if top_k & gt_set else 0.0

    # MRR@K (using max K = 10)
    max_k = max(k_values)
    mrr = 0.0
    for rank, idx in enumerate(retrieved_indices[:max_k], start=1):
        if idx in gt_set:
            mrr = 1.0 / rank
            break
    metrics["MRR@10"] = mrr

    return metrics


def aggregate_retrieval_metrics(
    all_retrieval_metrics: List[Dict[str, float]],
    all_categories: List[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate retrieval metrics across all queries, split by category.

    Args:
        all_retrieval_metrics: list of per-query metric dicts
        all_categories: list of category labels (optional)

    Returns:
        dict with "overall" and optional "category_N" keys
    """
    if not all_retrieval_metrics:
        return {}

    metric_names = list(all_retrieval_metrics[0].keys())

    results = {"overall": {}}
    category_aggregates = defaultdict(lambda: defaultdict(list))

    if all_categories is None:
        all_categories = [0] * len(all_retrieval_metrics)

    for metrics, cat in zip(all_retrieval_metrics, all_categories):
        for name in metric_names:
            category_aggregates[cat][name].append(metrics[name])

    # Overall
    for name in metric_names:
        values = [m[name] for m in all_retrieval_metrics]
        results["overall"][name] = {
            "mean": statistics.mean(values),
            "count": len(values),
        }

    # Per-category
    for cat in sorted(category_aggregates.keys()):
        results[f"category_{cat}"] = {}
        for name in metric_names:
            vals = category_aggregates[cat][name]
            results[f"category_{cat}"][name] = {
                "mean": statistics.mean(vals) if vals else 0.0,
                "count": len(vals),
            }

    return results
