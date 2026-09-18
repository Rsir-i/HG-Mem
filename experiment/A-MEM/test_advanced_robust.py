"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    # OpenAI
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json

    # Direct local model loading (e.g. Qwen-7B-Instruct)
    python test_advanced_robust.py --backend transformers \\
        --model /path/to/Qwen2.5-7B-Instruct \\
        --dataset data/locomo10.json

    # Small-sample validation (10% of dataset)
    python test_advanced_robust.py --backend transformers \\
        --model /path/to/Qwen2.5-7B-Instruct \\
        --dataset data/locomo10.json --ratio 0.1
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
import time
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics, calculate_retrieval_metrics, aggregate_retrieval_metrics
from datetime import datetime

# Download required NLTK data (punkt_tab for newer nltk, punkt as fallback)
for resource in ['punkt_tab', 'punkt', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass  # ignore if resource not available

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str,
                        force_k: Optional[int] = None) -> tuple:
        """Generate answer for a question — plain text, no JSON schema.

        Args:
            force_k: if provided, override self.retrieve_k for this call
                     (used for R@10-consistent generation evaluation)

        Returns:
            tuple: (response, user_prompt, raw_context, retrieved_indices)
        """
        k = force_k if force_k is not None else self.retrieve_k
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=k)
        context = raw_context

        # Also get indices for retrieval evaluation
        retrieved_indices = self.memory_system.retriever.search(keywords, k=k)
        if hasattr(retrieved_indices, 'tolist'):
            retrieved_indices = retrieved_indices.tolist()
        else:
            retrieved_indices = list(retrieved_indices) if retrieved_indices is not None else []

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context, retrieved_indices


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    # Pre-count total eligible QAs for progress bar
    allow_categories = [1, 2, 3, 4, 5]
    total_qa_target = sum(
        sum(1 for qa in s.qa if int(qa.category) in allow_categories)
        for s in samples
    )
    eval_logger.info(f"Total eligible QAs: {total_qa_target}")

    results = []
    all_metrics = []
    all_categories = []
    all_retrieval_metrics = []
    all_retrieval_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    total_elapsed = 0.0  # cumulative elapsed time (seconds) across all QAs
    r10_f1_values = []  # F1 values for R@10 Qwen-generated answers
    r10_f1_categories = []

    # Sanitize model name for directory usage (replace path separators)
    model_safe = model.replace("/", "_").replace("\\", "_").replace(":", "_")
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model_safe),
    )
    os.makedirs(memories_dir, exist_ok=True)

    # ============================ Phase 1: Build / Resume all memories ============================
    eval_logger.info("=" * 60)
    eval_logger.info("Phase 1: Building/resuming memories for all samples...")
    eval_logger.info("=" * 60)

    for sample_idx, sample in enumerate(samples):
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Sample {sample_idx}: cached memories exist, skipping build")
            # Load and warm up retriever so Phase 2 doesn't need to rebuild
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            eval_logger.info(f"Sample {sample_idx}: {len(cached_memories)} memories loaded")
        else:
            # Count total turns for progress bar
            all_turns = []
            for sess_id, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    all_turns.append((turns.date_time, turn))
            total_turns = len(all_turns)
            eval_logger.info(f"Sample {sample_idx}: no cache found, "
                             f"building {total_turns} new memories...")

            mem_t0 = time.time()
            pbar = tqdm(total=total_turns,
                        desc=f"Sample {sample_idx} memories",
                        unit="turn",
                        ncols=100)
            for idx, (turn_datatime, turn) in enumerate(all_turns, start=1):
                conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                agent.add_memory(conversation_tmp, time=turn_datatime)
                elapsed = time.time() - mem_t0
                avg = elapsed / idx
                pbar.set_postfix_str(f"avg {avg:.1f}s/turn")
                pbar.update(1)
            pbar.close()

            mem_elapsed = time.time() - mem_t0
            eval_logger.info(f"Sample {sample_idx}: memory build done in {mem_elapsed:.1f}s "
                             f"({mem_elapsed/total_turns:.1f}s/turn)")

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Sample {sample_idx}: cached {len(memories_to_cache)} memories")

        del agent  # free agent, model stays cached at class level

    eval_logger.info("Phase 1 complete: All memories built and cached.\n")

    # ============================ Phase 2: QA Evaluation ============================
    eval_logger.info("=" * 60)
    eval_logger.info("Phase 2: Evaluating all QAs...")
    eval_logger.info("=" * 60)

    qa_pbar = tqdm(total=total_qa_target, desc="QA Evaluation",
                   unit="q", ncols=120, position=0)

    for sample_idx, sample in enumerate(samples):
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        # ---- Build evidence -> memory_idx mapping for THIS sample ----
        evidence_to_memory_idx: Dict[str, int] = {}
        memory_idx = 0
        for session_id in sorted(sample.conversation.sessions.keys()):
            turns = sample.conversation.sessions[session_id]
            for turn_idx, turn in enumerate(turns.turns, start=1):
                evidence_key = f"D{session_id}:{turn_idx}"
                evidence_to_memory_idx[evidence_key] = memory_idx
                memory_idx += 1
        eval_logger.info(
            f"Sample {sample_idx}: evidence mapping with {len(evidence_to_memory_idx)} entries"
        )

        # Load cached memories (guaranteed to exist after Phase 1)
        with open(memory_cache_file, 'rb') as f:
            cached_memories = pickle.load(f)
        agent.memory_system.memories = cached_memories
        if os.path.exists(retriever_cache_file):
            agent.memory_system.retriever = agent.memory_system.retriever.load(
                retriever_cache_file, retriever_cache_embeddings_file
            )
        else:
            agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                cached_memories, 'all-MiniLM-L6-v2'
            )

        n_sample_qas = sum(1 for qa in sample.qa if int(qa.category) in allow_categories)
        eval_logger.info(f"Sample {sample_idx}: loaded memories, evaluating {n_sample_qas} QAs")

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                t_start = time.time()
                prediction, user_prompt, raw_context, retrieved_indices = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )
                t_elapsed = time.time() - t_start
                total_elapsed += t_elapsed

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                # ---- Retrieval metrics ----
                gt_indices = []
                for ev in qa.evidence:
                    ev = ev.strip()
                    if ev in evidence_to_memory_idx:
                        gt_indices.append(evidence_to_memory_idx[ev])
                ret_metrics = calculate_retrieval_metrics(
                    list(retrieved_indices), gt_indices
                )
                all_retrieval_metrics.append(ret_metrics)
                all_retrieval_categories.append(qa.category)

                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }

                # ---- R@10 Qwen Generation F1 ----
                if retrieve_k == 10:
                    r10_f1 = metrics["f1"]
                else:
                    r10_pred_raw, _, _, r10_indices = agent.answer_question(
                        qa.question, qa.category, qa.final_answer,
                        force_k=10,
                    )
                    r10_pred = parse_plain_text_answer(r10_pred_raw)
                    r10_f1 = calculate_metrics(r10_pred, qa.final_answer)["f1"] if qa.final_answer else 0.0
                r10_f1_values.append(r10_f1)
                r10_f1_categories.append(qa.category)

                all_metrics.append(metrics)

                # ---- Compact per-question log (one line, via tqdm.write to avoid messing up progress bar) ----
                r_str = (f"R@1={ret_metrics['R@1']:.0f} "
                         f"R@3={ret_metrics['R@3']:.0f} "
                         f"R@5={ret_metrics['R@5']:.0f} "
                         f"R@10={ret_metrics['R@10']:.0f} "
                         f"MRR={ret_metrics['MRR@10']:.3f}")
                tqdm.write(
                    f"[{t_elapsed:.1f}s] "
                    f"Q{total_questions} cat={qa.category} | "
                    f"pred='{prediction}' | ref='{qa.final_answer}' | "
                    f"F1={metrics['f1']:.3f} R10F1={r10_f1:.3f} | "
                    f"{r_str} | "
                    f"GT={gt_indices} top10={list(retrieved_indices[:10])}"
                )
                # Full prompt + context goes to log file for debugging
                eval_logger.info(f"  Q{total_questions}: {qa.question}")
                eval_logger.info(f"  User Prompt: {user_prompt}")
                eval_logger.info(f"  Raw Context: {raw_context}")
                all_categories.append(qa.category)

                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "retrieval_metrics": ret_metrics,
                    "r10_qwen_f1": r10_f1,
                    "elapsed_seconds": round(t_elapsed, 3),
                    "gt_evidence": qa.evidence,
                    "gt_memory_indices": gt_indices,
                    "retrieved_indices": list(retrieved_indices),
                }
                results.append(result)

                # Update progress bar
                avg_sofar = total_elapsed / total_questions
                qa_pbar.set_postfix_str(f"avg {avg_sofar:.1f}s/q  F1={metrics['f1']:.3f}")
                qa_pbar.update(1)

        del agent  # free agent for next sample

    qa_pbar.close()

    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    aggregate_retrieval = aggregate_retrieval_metrics(
        all_retrieval_metrics, all_retrieval_categories
    )

    # ---- R@10 Qwen F1 aggregate ----
    avg_r10_f1 = statistics.mean(r10_f1_values) if r10_f1_values else 0.0
    r10_f1_by_cat = defaultdict(list)
    for f1_val, cat in zip(r10_f1_values, r10_f1_categories):
        r10_f1_by_cat[cat].append(f1_val)

    # ---- Timing aggregate ----
    avg_elapsed = total_elapsed / total_questions if total_questions > 0 else 0.0

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "timing": {
            "total_seconds": round(total_elapsed, 2),
            "avg_seconds_per_question": round(avg_elapsed, 4),
        },
        "r10_qwen_f1": {
            "overall": round(avg_r10_f1, 4),
            "by_category": {str(c): round(statistics.mean(vals), 4)
                            for c, vals in sorted(r10_f1_by_cat.items())},
        },
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "aggregate_retrieval_metrics": aggregate_retrieval,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("=" * 60)
    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info(f"\n--- Timing ---")
    eval_logger.info(f"Total elapsed: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
    eval_logger.info(f"Average per question: {avg_elapsed:.2f}s")

    eval_logger.info("\n--- Generation Metrics ---")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    eval_logger.info("\n--- Retrieval Metrics ---")
    for split_name, ret_metrics in aggregate_retrieval.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in ret_metrics.items():
            if "mean" in stats:
                eval_logger.info(f"  {metric_name}: {stats['mean']:.4f} (n={stats['count']})")

    eval_logger.info(f"\n--- R@10 Qwen Generation F1 ---")
    eval_logger.info(f"Overall F1: {avg_r10_f1:.4f} (n={len(r10_f1_values)})")
    for cat in sorted(r10_f1_by_cat.keys()):
        vals = r10_f1_by_cat[cat]
        eval_logger.info(f"  Category {cat}: F1={statistics.mean(vals):.4f} (n={len(vals)})")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name or path (for transformers backend, supply the local model directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0, use e.g. 0.1 for small-sample validation)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, vllm, or transformers)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang/vLLM server host")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang/vLLM server port")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port,
    )


if __name__ == "__main__":
    main()
