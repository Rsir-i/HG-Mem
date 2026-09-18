# -*- coding: utf-8 -*-
"""
SeCom-style Evaluation Pipeline
================================
Implements topic-segmentation based memory construction + retrieval,
with turn-level evaluation metrics (R@1,3,5,10, MRR, F1).

Usage:
    python run_eval.py --model_path /path/to/Qwen-7B-Instruct --dataset locomo10
    python run_eval.py --model_path /path/to/Qwen-7B-Instruct --dataset longmemeval
    python run_eval.py --model_path /path/to/Qwen-7B-Instruct --dataset all
"""

import argparse
import json
import os
import re
import string
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===========================
#  Qwen-7B-Instruct Wrapper
# ===========================

class QwenLLM:
    """Local Qwen-7B-Instruct model loader (no vLLM, direct HF)."""

    def __init__(self, model_path: str):
        print(f"[QwenLLM] Loading model from {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[QwenLLM] Model loaded in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response


# ===========================
#  Data Structures
# ===========================

class Turn:
    """A single conversational turn."""

    __slots__ = ("uid", "speaker", "text", "session_idx", "turn_idx")

    def __init__(self, uid: str, speaker: str, text: str, session_idx: int, turn_idx: int):
        self.uid = uid
        self.speaker = speaker
        self.text = text
        self.session_idx = session_idx
        self.turn_idx = turn_idx

    def __repr__(self) -> str:
        return f"Turn({self.uid}, {self.speaker})"


class MemoryUnit:
    """A memory unit (segment/session) for retrieval."""

    __slots__ = ("idx", "turns", "page_content", "unit_type")

    def __init__(self, idx: int, turns: List[Turn], page_content: str, unit_type: str = "segment"):
        self.idx = idx
        self.turns = turns
        self.page_content = page_content
        self.unit_type = unit_type


# ===========================
#  Data Loaders
# ===========================

def load_locomo10(path: str) -> List[dict]:
    print(f"[Data] Loading locomo10 from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Data] {len(data)} dialogue samples loaded")
    return data


def load_longmemeval(path: str) -> List[dict]:
    print(f"[Data] Loading longmemeval from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Data] {len(data)} samples loaded")
    return data


def extract_sessions_locomo10(conversation: dict) -> List[List[dict]]:
    """Extract ordered sessions (list of turn dicts) from locomo10 conversation."""
    sessions = []
    i = 1
    while True:
        sess_key = f"session_{i}"
        if sess_key not in conversation:
            break
        session_data = conversation[sess_key]
        if isinstance(session_data, list) and len(session_data) > 0:
            sessions.append(session_data)
        i += 1
    return sessions


def build_turns_locomo10(conversation: dict) -> Tuple[List[Turn], Dict[str, Turn]]:
    """Build Turn objects and a uid->Turn map from locomo10 conversation."""
    turn_list: List[Turn] = []
    uid_map: Dict[str, Turn] = {}

    i = 1
    while True:
        sess_key = f"session_{i}"
        if sess_key not in conversation:
            break
        session_data = conversation[sess_key]
        if not isinstance(session_data, list):
            i += 1
            continue
        for t_idx, td in enumerate(session_data):
            uid = td.get("dia_id", "")
            speaker = td.get("speaker", "")
            text = td.get("text", "")
            turn = Turn(uid=uid, speaker=speaker, text=text, session_idx=i - 1, turn_idx=t_idx)
            turn_list.append(turn)
            if uid:
                uid_map[uid] = turn
        i += 1

    return turn_list, uid_map


def build_turns_longmemeval(haystack_sessions: List) -> Tuple[List[Turn], List[List[Turn]], Dict[int, List[Turn]]]:
    """Build turns from longmemeval haystack_sessions."""
    all_turns: List[Turn] = []
    session_turns: List[List[Turn]] = []  # turns grouped by session
    session_map: Dict[int, List[Turn]] = {}

    for s_idx, session in enumerate(haystack_sessions):
        st = []
        for t_idx, td in enumerate(session):
            role = td.get("role", "")
            content = td.get("content", "")
            uid = f"S{s_idx}-T{t_idx}"
            turn = Turn(uid=uid, speaker=role, text=content, session_idx=s_idx, turn_idx=t_idx)
            all_turns.append(turn)
            st.append(turn)
        session_turns.append(st)
        session_map[s_idx] = st

    return all_turns, session_turns, session_map


# ===========================
#  Segmentation (SeCom-style)
# ===========================

SEGMENT_PROMPT_TEMPLATE = """# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn conversation between a user and a chatbot into topically coherent units based on semantics. Successive user-bot exchanges with the same topic should be grouped into the same segmentation unit, and new segmentation units should be created when a topic shift occurs.

- **Data**: The input data is a series of user-bot exchanges separated by "\\n\\n". Each exchange consists of a single-turn conversation between the user and the chatbot, started with "[Exchange (Exchange Number)]: ".

## Requirements

### Output Format

- Output the segmentation results in **jsonl lines file** format. Each dictionary represents a segment. Each dictionary should include the following keys:
    - **segment_id**: The index of this segment, starting from 0.
    - **start_exchange_number**: The number of the **first** user-bot exchange in this segment.
    - **end_exchange_number**: The number of the **last** user-bot exchange in this segment.
    - **num_exchanges**: An integer indicating the number of user-bot exchanges in this segment, calculated as **end_exchange_number** - **start_exchange_number** + 1.

Here is an example of the expected output:
```
<segmentation>
{{"segment_id": 0, "start_exchange_number": 0, "end_exchange_number": 5, "num_exchanges": 6}}
{{"segment_id": 1, "start_exchange_number": 6, "end_exchange_number": 8, "num_exchanges": 3}}
</segmentation>
```

# Data

{text_to_be_segmented}

# Question

## Please generate the segmentation result from the input data that meets the following requirements:

- **No Missing Exchanges**: Ensure that the exchange numbers cover all exchanges in the given conversation without omission.
- **No Overlapping Exchanges**: Ensure that successive segments have no overlap in exchanges.
- **Accurate Counting**: The sum of **num_exchanges** across all segments should equal the total number of user-bot exchanges in the input.
- Provide your segmentation result between the tags: <segmentation></segmentation>.

# Output

Now, provide the segmentation result based on the instructions above."""


def segment_session(
    llm: QwenLLM,
    exchanges: List[str],
    session_idx: int,
    max_exchanges_per_batch: int = 15,
) -> List[List[int]]:
    """
    Segment a session into topic-coherent groups.
    Returns list of [start_idx, end_idx] boundaries.

    For sessions with many exchanges, splits into batches to stay within context limits.
    Falls back to fixed-size chunks if LLM segmentation fails.
    """
    n = len(exchanges)
    if n <= 1:
        return [[0, n - 1]]

    # For very long sessions, split into manageable batches
    if n <= max_exchanges_per_batch:
        return _segment_batch(llm, exchanges, session_idx, 0)
    else:
        # Split into overlapping batches and merge
        all_boundaries = []
        batch_start = 0
        while batch_start < n:
            batch_end = min(batch_start + max_exchanges_per_batch, n)
            batch_exchanges = exchanges[batch_start:batch_end]
            boundaries = _segment_batch(
                llm, batch_exchanges, session_idx, batch_start
            )
            all_boundaries.extend(
                [[b[0] + batch_start, b[1] + batch_start] for b in boundaries]
            )
            batch_start = batch_end

        # Merge consecutive boundaries if needed
        return _merge_boundaries(all_boundaries, n)


def _segment_batch(
    llm: QwenLLM,
    exchanges: List[str],
    session_idx: int,
    offset: int,
) -> List[List[int]]:
    """Segment a single batch of exchanges. Falls back to fixed windows on failure."""
    n = len(exchanges)
    if n <= 2:
        return [[0, n - 1]]

    # Build prompt
    exchanges_str = ""
    for i, ex in enumerate(exchanges):
        exchanges_str += f"[Exchange {offset + i}]: {ex}\n\n"

    prompt = SEGMENT_PROMPT_TEMPLATE.format(text_to_be_segmented=exchanges_str)

    try:
        response = llm.generate(prompt, max_tokens=512, temperature=0.1)
        boundaries = _parse_segmentation(response, n, offset)
        if boundaries:
            return boundaries
    except Exception:
        pass

    # Fallback: fixed-size chunks of ~3 exchanges
    chunk_size = 3
    boundaries = []
    for i in range(0, n, chunk_size):
        boundaries.append([i, min(i + chunk_size - 1, n - 1)])
    return boundaries


def _parse_segmentation(response: str, total_exchanges: int, offset: int) -> Optional[List[List[int]]]:
    """Parse the <segmentation>...</segmentation> output."""
    pattern = r"<segmentation>([\s\S]*?)</segmentation>"
    match = re.search(pattern, response)
    if not match:
        return None

    text = match.group(1).strip()
    lines = text.split("\n")
    boundaries = []
    for line in lines:
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_ex = int(obj.get("num_exchanges", 0))
        if n_ex <= 0:
            continue
        boundaries.append(n_ex)

    if not boundaries:
        return None

    # Verify total exchanges
    if sum(boundaries) != total_exchanges:
        return None

    # Convert to [start, end] pairs
    result = []
    start = 0
    for n_ex in boundaries:
        result.append([start, start + n_ex - 1])
        start += n_ex

    return result


def _merge_boundaries(
    boundaries: List[List[int]], total_n: int
) -> List[List[int]]:
    """Merge small segments that are adjacent."""
    if len(boundaries) <= 1:
        return boundaries

    merged = [boundaries[0]]
    for b in boundaries[1:]:
        prev = merged[-1]
        gap = b[0] - prev[1] - 1
        if gap <= 1:  # merge adjacent/slightly overlapping
            merged[-1] = [prev[0], b[1]]
        else:
            merged.append(b)
    return merged


# ===========================
#  Retrieval Engine
# ===========================

class RetrievalEngine:
    """FAISS-based dense retrieval with sentence-transformers."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Retrieval] Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.memory_units: List[MemoryUnit] = []
        self._dim = None

    def build_index(self, memory_units: List[MemoryUnit]):
        """Build FAISS index from memory units."""
        self.memory_units = memory_units
        texts = [mu.page_content for mu in memory_units]
        print(f"[Retrieval] Encoding {len(texts)} memory units ...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self._dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self._dim)  # inner product for normalized vectors
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 10) -> List[MemoryUnit]:
        """Retrieve top-k memory units for a query."""
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.memory_units)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.memory_units):
                results.append(self.memory_units[idx])
        return results


# ===========================
#  Evaluation Metrics
# ===========================

def normalize_answer(s: str) -> str:
    """Normalize text for F1 calculation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score for QA evaluation."""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_retrieval_metrics(
    all_retrieved_ids: List[List[str]],
    all_evidence_ids: List[Set[str]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute R@K and MRR at turn level."""
    if k_values is None:
        k_values = [1, 3, 5, 10]
    max_k = max(k_values)

    recall_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_scores: List[float] = []

    for retrieved_ids, evidence_set in zip(all_retrieved_ids, all_evidence_ids):
        if not evidence_set:
            for k in k_values:
                recall_scores[k].append(0.0)
            mrr_scores.append(0.0)
            continue

        # R@K: fraction of evidence turns found in top-K
        for k in k_values:
            top_k_ids: Set[str] = set(retrieved_ids[:k])
            hits = len(top_k_ids & evidence_set)
            recall_scores[k].append(hits / len(evidence_set))

        # MRR: 1 / rank of first relevant unit
        rank = None
        for i, rid in enumerate(retrieved_ids[:max_k]):
            if rid in evidence_set:
                rank = i + 1
                break
        mrr_scores.append(1.0 / rank if rank else 0.0)

    results = {}
    for k in k_values:
        results[f"R@{k}"] = float(np.mean(recall_scores[k]) * 100)
    results["MRR"] = float(np.mean(mrr_scores) * 100)
    return results


def compute_f1_metrics(
    predictions: List[str], ground_truths: List[str]
) -> Dict[str, float]:
    """Compute QA-F1 between predictions and ground truths."""
    f1_scores = []
    for pred, gt in zip(predictions, ground_truths):
        if isinstance(gt, list):
            score = max(qa_f1_score(pred, g) for g in gt)
        else:
            score = qa_f1_score(pred, str(gt))
        f1_scores.append(score)
    return {"qa_f1": float(np.mean(f1_scores) * 100)}


# ===========================
#  Answer Generation
# ===========================

GENERATION_PROMPT = """You are an intelligent assistant. You will be shown related conversation history (memory) relevant to the user's question. Please read the memory carefully and generate a concise, accurate answer.

{context}

Question: {question}

Answer:"""


def generate_answers(
    llm: QwenLLM,
    questions: List[str],
    contexts: List[str],
    batch_size: int = 1,
) -> Tuple[List[str], float]:
    """Generate answers using retrieved context. Returns (answers, total_time)."""
    answers = []
    total_time = 0.0
    for i, (q, ctx) in enumerate(
        tqdm(
            zip(questions, contexts),
            total=len(questions),
            desc="Generating answers",
        )
    ):
        prompt = GENERATION_PROMPT.format(context=ctx, question=q)
        t0 = time.time()
        ans = llm.generate(prompt, max_tokens=256, temperature=0.1)
        elapsed = time.time() - t0
        total_time += elapsed
        answers.append(ans.strip())
    return answers, total_time


# ===========================
#  Main Experiment Loops
# ===========================

def run_locomo10(
    llm: QwenLLM,
    retriever: RetrievalEngine,
    data: List[dict],
    use_segmentation: bool = True,
) -> dict:
    """Run experiment on locomo10 dataset."""
    results = {
        "dataset": "locomo10",
        "settings": {"segmentation": use_segmentation},
        "samples": [],
        "metrics": {},
        "timing": {},
    }

    all_retrieved_ids: List[List[str]] = []
    all_evidence_ids: List[Set[str]] = []
    all_predictions: List[str] = []
    all_ground_truths: List[str] = []

    total_preprocess_time = 0.0
    total_question_time = 0.0
    total_question_count = 0

    for sample_idx, sample in enumerate(
        tqdm(data, desc="Processing locomo10 samples")
    ):
        sample_result = {
            "sample_id": sample.get("sample_id", sample_idx),
            "qa_results": [],
        }

        # --- Step 1: Extract turns ---
        t0 = time.time()
        sessions = extract_sessions_locomo10(sample["conversation"])
        turn_list, uid_map = build_turns_locomo10(sample["conversation"])

        # --- Step 2: Segmentation (SeCom) ---
        if use_segmentation:
            memory_units = _segment_conversation_locomo10(llm, sessions, turn_list)
        else:
            # Fallback: each session = one memory unit
            memory_units = _session_as_units_locomo10(sessions, turn_list)

        # --- Step 3: Build retrieval index ---
        retriever.build_index(memory_units)

        preprocess_time = time.time() - t0
        total_preprocess_time += preprocess_time

        # --- Step 4: Process questions ---
        questions = []
        ground_truths_list = []
        evidence_ids_list: List[Set[str]] = []
        qa_pairs = sample.get("qa", [])

        for qa in qa_pairs:
            questions.append(qa["question"])
            ground_truths_list.append(qa["answer"])
            evidence_ids_list.append(set(qa.get("evidence", [])))

        contexts = []
        retrieved_ids_list: List[List[str]] = []

        q_start = time.time()
        for q_idx, q in enumerate(
            tqdm(questions, desc=f"Retrieving Q{sample_idx}", leave=False)
        ):
            retrieved_units = retriever.retrieve(q, top_k=10)

            # Extract turn UIDs from retrieved units (for retrieval metrics)
            retrieved_uids = []
            for mu in retrieved_units:
                for t in mu.turns:
                    if t.uid:
                        retrieved_uids.append(t.uid)
            retrieved_ids_list.append(retrieved_uids)

            # Build context for generation: concatenate turns in retrieved order
            context_parts = []
            for mu in retrieved_units:
                mu_text = "\n".join(
                    f"[{t.speaker}]: {t.text}" for t in mu.turns
                )
                context_parts.append(mu_text)
            contexts.append("\n\n".join(context_parts))

        # --- Step 5: Generate answers ---
        predictions, gen_time = generate_answers(llm, questions, contexts)
        question_time = (time.time() - q_start) + gen_time
        total_question_time += question_time
        total_question_count += len(questions)

        # --- Step 6: Store per-question results ---
        for q_idx, q in enumerate(questions):
            sample_result["qa_results"].append({
                "question": q,
                "ground_truth": ground_truths_list[q_idx],
                "prediction": predictions[q_idx],
                "evidence": list(evidence_ids_list[q_idx]),
            })

        # Accumulate for aggregate metrics
        all_retrieved_ids.extend(retrieved_ids_list)
        all_evidence_ids.extend(evidence_ids_list)
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths_list)

        results["samples"].append(sample_result)

    # --- Compute aggregate metrics ---
    retrieval_metrics = compute_retrieval_metrics(all_retrieved_ids, all_evidence_ids)
    f1_metrics = compute_f1_metrics(all_predictions, all_ground_truths)

    results["metrics"] = {**retrieval_metrics, **f1_metrics}
    results["timing"] = {
        "total_preprocess_seconds": round(total_preprocess_time, 2),
        "total_question_seconds": round(total_question_time, 2),
        "total_seconds": round(total_preprocess_time + total_question_time, 2),
        "avg_preprocess_per_sample_seconds": round(total_preprocess_time / len(data), 2),
        "avg_per_question_seconds": round(total_question_time / total_question_count, 2),
        "total_questions": total_question_count,
    }

    return results


def run_longmemeval(
    llm: QwenLLM,
    retriever: RetrievalEngine,
    data: List[dict],
) -> dict:
    """Run experiment on longmemeval dataset.
    For longmemeval, each session is independent (different topics),
    so we treat each session as a memory unit without segmentation.
    """
    results = {
        "dataset": "longmemeval",
        "settings": {"segmentation": False},
        "samples": [],
        "metrics": {},
        "timing": {},
    }

    all_retrieved_session_ids: List[List[str]] = []
    all_evidence_session_ids: List[Set[str]] = []
    all_predictions: List[str] = []
    all_ground_truths: List[str] = []

    total_preprocess_time = 0.0
    total_question_time = 0.0
    total_question_count = len(data)

    for sample_idx, sample in enumerate(
        tqdm(data, desc="Processing longmemeval samples")
    ):
        t0 = time.time()

        # --- Build memory units (each session = one unit) ---
        haystack_sessions = sample.get("haystack_sessions", [])
        _, session_turns_list, _ = build_turns_longmemeval(haystack_sessions)

        memory_units = []
        for s_idx, sturns in enumerate(session_turns_list):
            content = "\n".join(f"[{t.speaker}]: {t.text}" for t in sturns)
            mu = MemoryUnit(idx=s_idx, turns=sturns, page_content=content, unit_type="session")
            memory_units.append(mu)

        # --- Build FAISS index for this sample ---
        retriever.build_index(memory_units)

        preprocess_time = time.time() - t0
        total_preprocess_time += preprocess_time

        # --- Retrieve ---
        question = sample["question"]
        q_start = time.time()
        retrieved_units = retriever.retrieve(question, top_k=10)

        # Session-level IDs for retrieval metrics
        retrieved_session_ids = [str(mu.idx) for mu in retrieved_units]
        evidence_sessions = set(
            str(sid) for sid in sample.get("answer_session_ids", [])
        )

        # Build context
        context_parts = []
        for mu in retrieved_units:
            mu_text = "\n".join(f"[{t.speaker}]: {t.text}" for t in mu.turns)
            context_parts.append(mu_text)
        context = "\n\n".join(context_parts)

        # --- Generate answer ---
        prediction, gen_time = generate_answers(llm, [question], [context])
        prediction = prediction[0]
        question_time = (time.time() - q_start) + gen_time
        total_question_time += question_time

        # --- Store ---
        results["samples"].append({
            "question_id": sample.get("question_id", f"q_{sample_idx}"),
            "question": question,
            "ground_truth": sample["answer"],
            "prediction": prediction,
            "evidence_sessions": list(evidence_sessions),
        })

        all_retrieved_session_ids.append(retrieved_session_ids)
        all_evidence_session_ids.append(evidence_sessions)
        all_predictions.append(prediction)
        all_ground_truths.append(sample["answer"])

    # --- Compute metrics ---
    retrieval_metrics = compute_retrieval_metrics(
        all_retrieved_session_ids, all_evidence_session_ids
    )
    f1_metrics = compute_f1_metrics(all_predictions, all_ground_truths)

    results["metrics"] = {**retrieval_metrics, **f1_metrics}
    results["timing"] = {
        "total_preprocess_seconds": round(total_preprocess_time, 2),
        "total_question_seconds": round(total_question_time, 2),
        "total_seconds": round(total_preprocess_time + total_question_time, 2),
        "avg_preprocess_per_sample_seconds": round(total_preprocess_time / total_question_count, 2),
        "avg_per_question_seconds": round(total_question_time / total_question_count, 2),
        "total_questions": total_question_count,
    }

    return results


def _segment_conversation_locomo10(
    llm: QwenLLM,
    sessions: List[List[dict]],
    turn_list: List[Turn],
) -> List[MemoryUnit]:
    """Segment locomo10 conversation using SeCom-style topic segmentation."""
    memory_units: List[MemoryUnit] = []
    mu_idx = 0
    turn_ptr = 0

    for sess_idx, session_data in enumerate(
        tqdm(sessions, desc="Segmenting sessions", leave=False)
    ):
        # Prepare exchanges
        exchanges = []
        for td in session_data:
            text = td.get("text", "")
            speaker = td.get("speaker", "")
            exchanges.append(f"{speaker}: {text}")

        if not exchanges:
            continue

        # Segment this session
        boundaries = segment_session(llm, exchanges, sess_idx)

        # Create memory units from segments
        for seg_start, seg_end in boundaries:
            turns_in_segment = turn_list[turn_ptr + seg_start : turn_ptr + seg_end + 1]
            content = "\n".join(
                f"[{t.speaker}]: {t.text}" for t in turns_in_segment
            )
            mu = MemoryUnit(idx=mu_idx, turns=turns_in_segment, page_content=content, unit_type="segment")
            memory_units.append(mu)
            mu_idx += 1

        turn_ptr += len(exchanges)

    return memory_units


def _session_as_units_locomo10(
    sessions: List[List[dict]],
    turn_list: List[Turn],
) -> List[MemoryUnit]:
    """Fallback: each session is one memory unit."""
    memory_units: List[MemoryUnit] = []
    turn_ptr = 0
    for sess_idx, session_data in enumerate(sessions):
        n = len(session_data)
        turns_in_session = turn_list[turn_ptr : turn_ptr + n]
        content = "\n".join(
            f"[{t.speaker}]: {t.text}" for t in turns_in_session
        )
        mu = MemoryUnit(
            idx=sess_idx, turns=turns_in_session, page_content=content, unit_type="session"
        )
        memory_units.append(mu)
        turn_ptr += n
    return memory_units


# ===========================
#  Entry Point
# ===========================

def main():
    parser = argparse.ArgumentParser(description="SeCom Evaluation Pipeline")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen-7B-Instruct model")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["locomo10", "longmemeval", "all"],
                        help="Which dataset to evaluate")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing dataset files (default: ../example/)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--no_segmentation", action="store_true",
                        help="Skip topic segmentation (use session-level units)")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model for retrieval")
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "..", "example")
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    locomo10_path = os.path.join(data_dir, "locomo10.json")
    longmemeval_path = os.path.join(data_dir, "longmemeval_s_cleaned.json")

    # Initialize models
    print("=" * 60)
    print("SeCom Evaluation Pipeline")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Segmentation: {not args.no_segmentation}")
    print("=" * 60)

    llm = QwenLLM(args.model_path)
    retriever = RetrievalEngine(embedding_model_name=args.embedding_model)

    datasets_to_run = []
    if args.dataset in ("locomo10", "all"):
        datasets_to_run.append("locomo10")
    if args.dataset in ("longmemeval", "all"):
        datasets_to_run.append("longmemeval")

    for ds_name in datasets_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running {ds_name}")
        print(f"{'=' * 60}")

        overall_t0 = time.time()

        if ds_name == "locomo10":
            data = load_locomo10(locomo10_path)
            result = run_locomo10(
                llm, retriever, data,
                use_segmentation=not args.no_segmentation,
            )
            output_file = os.path.join(output_dir, "locomo10_results.json")

        elif ds_name == "longmemeval":
            data = load_longmemeval(longmemeval_path)
            result = run_longmemeval(llm, retriever, data)
            output_file = os.path.join(output_dir, "longmemeval_results.json")

        overall_time = time.time() - overall_t0
        result["timing"]["overall_wall_seconds"] = round(overall_time, 2)

        # Print summary
        print(f"\n{'=' * 40}")
        print(f"Results for {ds_name}")
        print(f"{'=' * 40}")
        print("\n[Retrieval Metrics]")
        for k, v in result["metrics"].items():
            if k.startswith("R@") or k == "MRR":
                print(f"  {k}: {v:.2f}")
        print("\n[Generation Metrics]")
        for k, v in result["metrics"].items():
            if k not in ("MRR",) and not k.startswith("R@"):
                print(f"  {k}: {v:.2f}")
        print("\n[Timing]")
        for k, v in result["timing"].items():
            print(f"  {k}: {v}")
        print(f"\nSaving to: {output_file}")

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
