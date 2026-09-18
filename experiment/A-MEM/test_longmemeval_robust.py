"""
Evaluation harness for LongMemEval dataset using the robust memory layer (no JSON schema dependency).

Dataset: longmemeval_s_cleaned.json
- 500 samples, each = 1 question + ~53 haystack sessions
- Memory granularity: 1 memory per session
- Ground truth for retrieval: answer_session_ids position in haystack

Usage:
    # OpenAI
    python test_longmemeval_robust.py --backend openai --model gpt-4o-mini --dataset data/longmemeval_s_cleaned.json

    # Direct local model loading (e.g. Qwen-7B-Instruct)
    python test_longmemeval_robust.py --backend transformers \\
        --model /path/to/Qwen2.5-7B-Instruct \\
        --dataset data/longmemeval_s_cleaned.json

    # Small-sample validation (10% of dataset)
    python test_longmemeval_robust.py --backend transformers \\
        --model /path/to/Qwen2.5-7B-Instruct \\
        --dataset data/longmemeval_s_cleaned.json --ratio 0.1
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
import nltk
from sentence_transformers import SentenceTransformer
import statistics
from collections import defaultdict
import pickle
import time
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics, calculate_retrieval_metrics, aggregate_retrieval_metrics, set_sbert_model_path
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

logger = logging.getLogger("longmemeval_robust")


# ──────────────────────────── Data Loader ────────────────────────────

def load_longmemeval_dataset(file_path: str) -> List[Dict]:
    """
    Load LongMemEval dataset from JSON file.
    Each sample: {question_id, question_type, question, question_date,
                  answer, answer_session_ids, haystack_dates,
                  haystack_session_ids, haystack_sessions}
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Pre-compute ground-truth indices for each sample
    for sample in data:
        hs_ids = sample['haystack_session_ids']
        ans_ids = set(sample['answer_session_ids'])
        sample['_gt_indices'] = [i for i, sid in enumerate(hs_ids) if sid in ans_ids]

    print(f"Loaded {len(data)} samples from {file_path}")
    return data


# ──────────────────────────── Agent ────────────────────────────

class LongMemEvalAgent:
    """Agent using the robust memory system for LongMemEval."""

    def __init__(self, model, backend, retrieve_k,
                 sglang_host="http://localhost", sglang_port=30000,
                 embed_model='all-MiniLM-L6-v2'):
        self.embed_model = embed_model
        self.memory_system = RobustAgenticMemorySystem(
            model_name=embed_model,
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

    def build_memories_from_sessions(self, haystack_sessions: List,
                                     haystack_session_ids: List[str],
                                     haystack_dates: Optional[List] = None):
        """
        Build memory entries from haystack sessions.
        Each session becomes one memory entry with concatenated turns.
        """
        for i, session_turns in enumerate(haystack_sessions):
            # Build session text from turns
            lines = []
            session_id = haystack_session_ids[i] if i < len(haystack_session_ids) else f"session_{i}"
            lines.append(f"Session {session_id}:")

            if haystack_dates and i < len(haystack_dates):
                lines.append(f"Date: {haystack_dates[i]}")

            for turn in session_turns:
                role = turn.get("role", "unknown").capitalize()
                content = turn.get("content", "")
                lines.append(f"{role}: {content}")

            session_text = "\n".join(lines)

            # Use session date as time if available
            session_time = haystack_dates[i] if haystack_dates and i < len(haystack_dates) else None
            self.memory_system.add_note(session_text, time=session_time)

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

    def answer_question(self, question: str, force_k: Optional[int] = None) -> tuple:
        """
        Generate answer for a question.

        Args:
            force_k: if provided, override self.retrieve_k for this call
                     (used for R@10-consistent generation evaluation)

        Returns:
            tuple: (response, user_prompt, raw_context, retrieved_indices, timing_dict)
            timing_dict keys: t_keywords, t_retrieve, t_retrieve_idx, t_generate, t_total
        """
        timing = {}
        t_total_start = time.time()

        k = force_k if force_k is not None else self.retrieve_k

        t0 = time.time()
        keywords = self.generate_query_llm(question)
        timing['t_keywords'] = time.time() - t0

        t0 = time.time()
        raw_context = self.retrieve_memory(keywords, k=k)
        timing['t_retrieve'] = time.time() - t0

        context = raw_context

        # Get indices for retrieval evaluation
        t0 = time.time()
        retrieved_indices = self.memory_system.retriever.search(keywords, k=k)
        if hasattr(retrieved_indices, 'tolist'):
            retrieved_indices = retrieved_indices.tolist()
        else:
            retrieved_indices = list(retrieved_indices) if retrieved_indices is not None else []
        timing['t_retrieve_idx'] = time.time() - t0

        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""

        t0 = time.time()
        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=0.7,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        timing['t_generate'] = time.time() - t0
        timing['t_total'] = time.time() - t_total_start

        return response, user_prompt, raw_context, retrieved_indices, timing


# ──────────────────────────── Logging ────────────────────────────

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('longmemeval_eval')
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


# ──────────────────────────── Turn-Level Retrieval ────────────────────────────

def _turn_level_retrieval(embed_model, sample, retrieved_session_indices, query, top_k=10):
    """
    Within top-1 retrieved session, rank individual turns by embedding similarity to the query.

    Returns:
        (turn_ret_metrics, top_turn_texts) where:
        - turn_ret_metrics: dict with R@1/3/5/10, MRR@10 at turn level
        - top_turn_texts: text of top-ranked turns (for logging)
    """
    haystack_sessions = sample['haystack_sessions']

    # Collect all turns from retrieved sessions
    candidate_turns = []
    for mem_idx in retrieved_session_indices:
        if mem_idx < len(haystack_sessions):
            for turn in haystack_sessions[mem_idx]:
                candidate_turns.append({
                    'text': turn.get('content', ''),
                    'is_answer': turn.get('has_answer', False),
                })

    n_candidates = len(candidate_turns)
    if n_candidates == 0:
        empty_metrics = {'R@1': 0.0, 'R@3': 0.0, 'R@5': 0.0, 'R@10': 0.0, 'MRR@10': 0.0}
        return empty_metrics, []

    # Embed turns and query
    turn_texts = [t['text'] for t in candidate_turns]
    turn_embs = embed_model.encode(turn_texts, convert_to_numpy=True)
    query_emb = embed_model.encode([query], convert_to_numpy=True)

    # Cosine similarity (normalized dot product)
    turn_embs = turn_embs / (np.linalg.norm(turn_embs, axis=1, keepdims=True) + 1e-10)
    query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    sims = np.dot(query_emb, turn_embs.T)[0]

    # Rank descending by similarity
    ranked = np.argsort(-sims)

    # Top-k turn indices in candidate pool
    top_indices = ranked[:min(top_k, n_candidates)].tolist()

    # GT: indices of turns with has_answer=true within candidate pool
    gt_indices = [i for i, t in enumerate(candidate_turns) if t['is_answer']]

    # Turn-level retrieval metrics
    turn_ret_metrics = calculate_retrieval_metrics(top_indices, gt_indices)

    # Top turn texts for logging
    top_texts = [candidate_turns[i]['text'][:80] for i in top_indices]

    return turn_ret_metrics, top_texts


# ──────────────────────────── Main Evaluation ────────────────────────────

def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     embed_model: str = "all-MiniLM-L6-v2"):
    """Evaluate the robust agent on the LongMemEval dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_longmemeval_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
    eval_logger.info(f"Embedding model: {embed_model}")
    print("  -> Loading embedding model...", flush=True)
    set_sbert_model_path(embed_model)

    print("  -> Loading dataset...", flush=True)
    samples = load_longmemeval_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    total_qa_target = len(samples)
    eval_logger.info(f"Total questions to evaluate: {total_qa_target}")

    # Sanitize model name for directory usage
    model_safe = model.replace("/", "_").replace("\\", "_").replace(":", "_")
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_longmemeval_{}_{}".format(backend, model_safe),
    )
    os.makedirs(memories_dir, exist_ok=True)

    # ============================ Phase 1: Build / Resume all memories ============================
    eval_logger.info("=" * 60)
    eval_logger.info("Phase 1: Building/resuming memories for all samples...")
    eval_logger.info("=" * 60)
    print("[2/3] Building/resuming memories (Phase 1)...", flush=True)

    total_build_time = 0.0   # cumulative build time across all samples
    total_build_sessions = 0  # total sessions built

    for sample_idx, sample in enumerate(samples):
        agent = LongMemEvalAgent(model, backend, retrieve_k,
                                 sglang_host, sglang_port,
                                 embed_model=embed_model)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        num_sessions = len(sample['haystack_sessions'])
        eval_logger.info(
            f"Sample {sample_idx}: type={sample['question_type']}, "
            f"{num_sessions} sessions, {len(sample['_gt_indices'])} answer sessions"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Sample {sample_idx}: cached memories exist, loading...")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, embed_model
                )
            eval_logger.info(f"Sample {sample_idx}: {len(cached_memories)} memories loaded")
        else:
            eval_logger.info(f"Sample {sample_idx}: building {num_sessions} new memories...")

            mem_t0 = time.time()
            pbar = tqdm(total=num_sessions,
                        desc=f"Sample {sample_idx} memories",
                        unit="sess",
                        ncols=100)
            # Build memories session by session
            for i, session_turns in enumerate(sample['haystack_sessions']):
                lines = []
                session_id = sample['haystack_session_ids'][i]
                lines.append(f"Session {session_id}:")

                if sample.get('haystack_dates') and i < len(sample['haystack_dates']):
                    lines.append(f"Date: {sample['haystack_dates'][i]}")

                for turn in session_turns:
                    role = turn.get("role", "unknown").capitalize()
                    content = turn.get("content", "")
                    lines.append(f"{role}: {content}")

                session_text = "\n".join(lines)
                session_time = (
                    sample['haystack_dates'][i]
                    if sample.get('haystack_dates') and i < len(sample['haystack_dates'])
                    else None
                )
                agent.memory_system.add_note(session_text, time=session_time)

                elapsed = time.time() - mem_t0
                avg = elapsed / (i + 1)
                pbar.set_postfix_str(f"avg {avg:.1f}s/sess")
                pbar.update(1)
            pbar.close()

            mem_elapsed = time.time() - mem_t0
            total_build_time += mem_elapsed
            total_build_sessions += num_sessions
            eval_logger.info(f"Sample {sample_idx}: memory build done in {mem_elapsed:.1f}s "
                             f"({mem_elapsed/num_sessions:.1f}s/session)")

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Sample {sample_idx}: cached {len(memories_to_cache)} memories")

        del agent  # free agent, model stays cached at class level

    eval_logger.info("Phase 1 complete: All memories built and cached.\n")
    eval_logger.info(f"--- Phase 1 Build Summary ---")
    eval_logger.info(f"Total build time: {total_build_time:.2f}s ({total_build_time/60:.2f}min)")
    if total_build_sessions > 0:
        eval_logger.info(f"Total sessions built: {total_build_sessions}")
        eval_logger.info(f"Average per session: {total_build_time/total_build_sessions:.2f}s")
        eval_logger.info(f"Average per sample: {total_build_time/len(samples):.2f}s")
    eval_logger.info("")

    # ============================ Phase 2: QA Evaluation ============================
    eval_logger.info("=" * 60)
    eval_logger.info("Phase 2: Evaluating all questions...")
    eval_logger.info("=" * 60)
    print("[3/3] Running QA evaluation (Phase 2)...", flush=True)

    qa_pbar = tqdm(total=total_qa_target, desc="QA Evaluation",
                   unit="q", ncols=120, position=0)

    results = []
    all_metrics = []
    all_types = []  # question_type per question
    all_retrieval_metrics = []
    all_retrieval_types = []
    total_questions = 0
    type_counts = defaultdict(int)
    total_elapsed = 0.0  # cumulative elapsed time (seconds) across all QAs
    total_t_keywords = 0.0   # cumulative keyword-generation time
    total_t_retrieve = 0.0   # cumulative memory-retrieval time
    total_t_retrieve_idx = 0.0  # cumulative retrieval-index search time
    total_t_generate = 0.0   # cumulative answer-generation time
    r10_f1_values = []
    r10_f1_types = []

    # Turn-level retrieval: metrics per question
    all_turn_retrieval_metrics = []
    all_turn_retrieval_types = []

    # Load embedding model for turn-level re-ranking
    print("  -> Loading turn-level embedding model...", flush=True)
    try:
        turn_embed_model = SentenceTransformer(embed_model)
        print("  -> Turn embed model loaded.", flush=True)
    except Exception as e:
        print(f"  -> Warning: Could not load turn embed model: {e}", flush=True)
        turn_embed_model = None

    for sample_idx, sample in enumerate(samples):
        question_type = sample['question_type']
        question = sample['question']
        answer = sample['answer']
        gt_indices = sample['_gt_indices']  # pre-computed ground-truth memory indices

        agent = LongMemEvalAgent(model, backend, retrieve_k,
                                 sglang_host, sglang_port,
                                 embed_model=embed_model)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
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
                cached_memories, embed_model
            )

        # ── Evaluate the single question ──
        total_questions += 1
        type_counts[question_type] += 1

        prediction, user_prompt, raw_context, retrieved_indices, q_timing = agent.answer_question(question)
        t_elapsed = q_timing['t_total']
        total_elapsed += t_elapsed
        total_t_keywords += q_timing['t_keywords']
        total_t_retrieve += q_timing['t_retrieve']
        total_t_retrieve_idx += q_timing['t_retrieve_idx']
        total_t_generate += q_timing['t_generate']

        # Parse the prediction
        prediction = parse_plain_text_answer(prediction)

        # ---- Session-level retrieval metrics ----
        ret_metrics = calculate_retrieval_metrics(
            list(retrieved_indices), gt_indices
        )
        all_retrieval_metrics.append(ret_metrics)
        all_retrieval_types.append(question_type)

        # ---- Turn-level retrieval (within retrieved sessions) ----
        if turn_embed_model is not None:
            turn_ret, top_turn_texts = _turn_level_retrieval(
                turn_embed_model, sample, list(retrieved_indices[:1]),
                question, top_k=10,
            )
        else:
            turn_ret = {'R@1': 0.0, 'R@3': 0.0, 'R@5': 0.0, 'R@10': 0.0, 'MRR@10': 0.0}
            top_turn_texts = []
        all_turn_retrieval_metrics.append(turn_ret)
        all_turn_retrieval_types.append(question_type)

        # ---- Generation metrics ----
        metrics = calculate_metrics(prediction, answer) if answer else {
            "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
            "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
            "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
        }

        # ---- R@10 Qwen Generation F1 ----
        if retrieve_k == 10:
            r10_f1 = metrics["f1"]
        else:
            r10_pred_raw, _, _, _, _ = agent.answer_question(question, force_k=10)
            r10_pred = parse_plain_text_answer(r10_pred_raw)
            r10_f1 = calculate_metrics(r10_pred, answer)["f1"] if answer else 0.0
        r10_f1_values.append(r10_f1)
        r10_f1_types.append(question_type)

        all_metrics.append(metrics)

        # ---- Compact per-question log ----
        r_str = (f"R@1={ret_metrics['R@1']:.0f} "
                 f"R@3={ret_metrics['R@3']:.0f} "
                 f"R@5={ret_metrics['R@5']:.0f} "
                 f"R@10={ret_metrics['R@10']:.0f} "
                 f"MRR={ret_metrics['MRR@10']:.3f}")
        tr_str = (f"tR@1={turn_ret['R@1']:.0f} "
                  f"tR@3={turn_ret['R@3']:.0f} "
                  f"tR@5={turn_ret['R@5']:.0f} "
                  f"tR@10={turn_ret['R@10']:.0f} "
                  f"tMRR={turn_ret['MRR@10']:.3f}")
        tqdm.write(
            f"[{t_elapsed:.1f}s kw={q_timing['t_keywords']:.1f}s "
            f"ret={q_timing['t_retrieve']:.2f}s "
            f"gen={q_timing['t_generate']:.1f}s] "
            f"Q{total_questions} type={question_type} | "
            f"pred='{prediction}' | ref='{answer}' | "
            f"F1={metrics['f1']:.3f} R10F1={r10_f1:.3f} | "
            f"{r_str} | {tr_str} | "
            f"GT={gt_indices} top10={list(retrieved_indices[:10])}"
        )
        eval_logger.info(f"  Q{total_questions}: {question}")
        eval_logger.info(f"  User Prompt: {user_prompt}")
        eval_logger.info(f"  Raw Context: {raw_context}")
        all_types.append(question_type)

        result = {
            "sample_id": sample_idx,
            "question_id": sample.get("question_id", str(sample_idx)),
            "question_type": question_type,
            "question": question,
            "prediction": prediction,
            "reference": answer,
            "metrics": metrics,
            "retrieval_metrics": ret_metrics,
            "r10_qwen_f1": r10_f1,
            "elapsed_seconds": round(t_elapsed, 3),
            "timing_breakdown": {
                "t_keywords": round(q_timing['t_keywords'], 4),
                "t_retrieve": round(q_timing['t_retrieve'], 4),
                "t_retrieve_idx": round(q_timing['t_retrieve_idx'], 4),
                "t_generate": round(q_timing['t_generate'], 4),
                "t_total": round(q_timing['t_total'], 4),
            },
            "gt_indices": gt_indices,
            "retrieved_indices": list(retrieved_indices),
            "turn_retrieval_metrics": turn_ret,
        }
        results.append(result)

        # Update progress bar
        avg_sofar = total_elapsed / total_questions
        qa_pbar.set_postfix_str(f"avg {avg_sofar:.1f}s/q  F1={metrics['f1']:.3f}")
        qa_pbar.update(1)

        del agent

    qa_pbar.close()

    # ── Aggregation ──
    aggregate_results = aggregate_metrics(all_metrics, all_types)
    aggregate_retrieval = aggregate_retrieval_metrics(
        all_retrieval_metrics, all_retrieval_types
    )
    aggregate_turn_retrieval = aggregate_retrieval_metrics(
        all_turn_retrieval_metrics, all_turn_retrieval_types
    )

    # R@10 F1 aggregate by question type
    avg_r10_f1 = statistics.mean(r10_f1_values) if r10_f1_values else 0.0
    r10_f1_by_type = defaultdict(list)
    for f1_val, qtype in zip(r10_f1_values, r10_f1_types):
        r10_f1_by_type[qtype].append(f1_val)

    # Timing aggregate
    avg_elapsed = total_elapsed / total_questions if total_questions > 0 else 0.0
    avg_t_keywords = total_t_keywords / total_questions if total_questions > 0 else 0.0
    avg_t_retrieve = total_t_retrieve / total_questions if total_questions > 0 else 0.0
    avg_t_retrieve_idx = total_t_retrieve_idx / total_questions if total_questions > 0 else 0.0
    avg_t_generate = total_t_generate / total_questions if total_questions > 0 else 0.0

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "timing": {
            "total_seconds": round(total_elapsed, 2),
            "avg_seconds_per_question": round(avg_elapsed, 4),
            "breakdown_avg_seconds": {
                "t_keywords": round(avg_t_keywords, 4),
                "t_retrieve": round(avg_t_retrieve, 4),
                "t_retrieve_idx": round(avg_t_retrieve_idx, 4),
                "t_generate": round(avg_t_generate, 4),
            },
            "phase1_build_seconds": round(total_build_time, 2),
            "phase1_build_sessions": total_build_sessions,
        },
        "r10_qwen_f1": {
            "overall": round(avg_r10_f1, 4),
            "by_type": {str(t): round(statistics.mean(vals), 4)
                        for t, vals in sorted(r10_f1_by_type.items())},
        },
        "type_distribution": dict(type_counts),
        "aggregate_metrics": aggregate_results,
        "aggregate_retrieval_metrics": aggregate_retrieval,
        "aggregate_turn_retrieval_metrics": aggregate_turn_retrieval,
        "individual_results": results,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    # ── Summary Logging ──
    eval_logger.info("=" * 60)
    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Type Distribution:")
    for qtype, count in sorted(type_counts.items()):
        eval_logger.info(f"  {qtype}: {count} ({count/total_questions*100:.1f}%)")

    eval_logger.info(f"\n--- Timing ---")
    eval_logger.info(f"Phase 1 Build total: {total_build_time:.2f}s ({total_build_time/60:.2f}min)"
                     f" | sessions={total_build_sessions}")
    eval_logger.info(f"Phase 2 QA total: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
    eval_logger.info(f"Average per question: {avg_elapsed:.2f}s")
    eval_logger.info(f"  - Keywords generation (avg): {avg_t_keywords:.2f}s")
    eval_logger.info(f"  - Memory retrieval   (avg): {avg_t_retrieve:.3f}s")
    eval_logger.info(f"  - Retrieve idx search(avg): {avg_t_retrieve_idx:.3f}s")
    eval_logger.info(f"  - Answer generation  (avg): {avg_t_generate:.2f}s")

    eval_logger.info("\n--- Generation Metrics ---")
    for split_name, split_metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in split_metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    eval_logger.info("\n--- Turn-Level Retrieval Metrics ---")
    for split_name, ret_metrics in aggregate_turn_retrieval.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in ret_metrics.items():
            if "mean" in stats:
                eval_logger.info(f"  {metric_name}: {stats['mean']:.4f} (n={stats['count']})")

    eval_logger.info(f"\n--- R@10 Qwen Generation F1 ---")
    eval_logger.info(f"Overall F1: {avg_r10_f1:.4f} (n={len(r10_f1_values)})")
    for qtype in sorted(r10_f1_by_type.keys()):
        vals = r10_f1_by_type[qtype]
        eval_logger.info(f"  {qtype}: F1={statistics.mean(vals):.4f} (n={len(vals)})")

    # ── Summary Table ──
    _print_summary_table(
        aggregate_turn_retrieval,
        aggregate_results, type_counts,
        avg_elapsed, total_questions,
        eval_logger,
    )

    return final_results


def _print_summary_table(aggregate_turn_retrieval, aggregate_results,
                         type_counts, avg_elapsed, total_questions, eval_logger):
    """Print a single summary table: turn-level retrieval + F1."""
    type_order = sorted(type_counts.keys(),
                        key=lambda t: type_counts[t], reverse=True)

    # ── Table: Turn-Level Retrieval + F1 ──
    header = f"{'Type':<26s} {'n':>5s} {'tR@1':>7s} {'tR@3':>7s} {'tR@5':>7s} {'tR@10':>7s} {'tMRR':>7s} {'F1':>7s}"
    sep = "-" * len(header)

    lines = []
    lines.append("")
    lines.append("=" * len(header))
    lines.append("  TURN-LEVEL RETRIEVAL (within top-1 session) + F1")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    for qtype in type_order:
        ret = aggregate_turn_retrieval.get(qtype, {})
        gen = aggregate_results.get(qtype, {})
        n = type_counts[qtype]
        r1 = ret.get("R@1", {}).get("mean", 0.0)
        r3 = ret.get("R@3", {}).get("mean", 0.0)
        r5 = ret.get("R@5", {}).get("mean", 0.0)
        r10 = ret.get("R@10", {}).get("mean", 0.0)
        mrr = ret.get("MRR@10", {}).get("mean", 0.0)
        f1v = gen.get("f1", {}).get("mean", 0.0)
        lines.append(
            f"{qtype:<26s} {n:>5d} {r1:>6.4f} {r3:>6.4f} {r5:>6.4f} {r10:>6.4f} {mrr:>6.4f} {f1v:>6.4f}"
        )

    ret = aggregate_turn_retrieval.get("overall", {})
    gen = aggregate_results.get("overall", {})
    r1 = ret.get("R@1", {}).get("mean", 0.0)
    r3 = ret.get("R@3", {}).get("mean", 0.0)
    r5 = ret.get("R@5", {}).get("mean", 0.0)
    r10 = ret.get("R@10", {}).get("mean", 0.0)
    mrr = ret.get("MRR@10", {}).get("mean", 0.0)
    f1v = gen.get("f1", {}).get("mean", 0.0)

    lines.append(sep)
    lines.append(
        f"{'OVERALL':<26s} {total_questions:>5d} {r1:>6.4f} {r3:>6.4f} {r5:>6.4f} {r10:>6.4f} {mrr:>6.4f} {f1v:>6.4f}"
    )
    lines.append(sep)
    lines.append(f"Avg time per question: {avg_elapsed:.2f}s")
    lines.append("")

    for line in lines:
        eval_logger.info(line)
    # Also print to stdout so it's visible in nohup output
    for line in lines:
        print(line, flush=True)


# ──────────────────────────── CLI ────────────────────────────

def main():
    import sys
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LongMemEval dataset"
    )
    parser.add_argument("--dataset", type=str, default="data/longmemeval_s_cleaned.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name or path (for transformers backend, supply the local model directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0, use e.g. 0.1 for small-sample validation)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, vllm, or transformers)")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang/vLLM server host")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang/vLLM server port")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2",
                        help="Path or name of SentenceTransformer embedding model")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    print(f"[1/3] Loading LLM model ({args.model}) + embedding model ({args.embed_model})...", flush=True)
    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.retrieve_k,
        args.sglang_host, args.sglang_port,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
