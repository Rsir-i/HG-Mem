"""
LongMemEval experiment: three-path hybrid recall + graph-signal re-ranking
============================================================================
Overview
--------
HG-Mem on the LongMemEval (`longmemeval_s_cleaned`) benchmark. The pipeline
is the same as the LoCoMo script:

1. Build a hierarchical memory graph per question (Session -> Topic-Cluster
   -> Turn) from the haystack of every question.
2. Recall candidate turns through three complementary paths (BM25,
   MiniLM->session, MiniLM->topic cluster).
3. Merge & de-duplicate the candidates, then re-rank them using three
   normalised signals (BM25 lexical, turn-level semantic, topic-aware
   semantic) to produce the final top-*k* turns.

Notes on the dataset
--------------------
The released dataset ships with the `has_answer` flag inside every haystack
message. We treat every message with `has_answer == true` as a gold-evidence
turn. The turn ID is constructed as `f"{session_idx}_{msg_idx}"`.

Compared with the LoCoMo script, this script additionally generates session
summaries with Qwen2.5-7B-Instruct, because LongMemEval does not provide
session summaries in the released JSON.

Three recall paths
------------------
  Path 1: BM25(query, turn)              top-N      (lexical backbone)
  Path 2: MiniLM(query, session_summary)  top-K      -> all turns
  Path 3: MiniLM(query, topic_cluster)    top-K      -> only turns inside
                                                  the cluster

Baselines (turn-level)
----------------------
  - BM25
  - MiniLM (all-MiniLM-L6-v2 cosine similarity)
  - BM25 + MiniLM with Reciprocal Rank Fusion (RRF)
"""

import argparse
import json, sys, io, math, os, time, re
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


# ============================================================================
# Tokenisation
# ============================================================================
def _ensure_nltk():
    """Make sure NLTK and the stopword corpus are available."""
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

_STOPS = _ensure_nltk()

def tokenize(text, stem=False):
    """Lower-case, strip punctuation, drop stop-words and 1-char tokens."""
    t = text.lower()
    for ch in '?!.,;:\"\'()[]{}-\n\r':
        t = t.replace(ch, ' ')
    tokens = [w for w in t.split() if w not in _STOPS and len(w) > 1]
    if stem:
        # Lightweight Porter stemmer (S-removal + ly/ing/ed only).
        out = []
        for w in tokens:
            if w.endswith('ies') and len(w) > 4:
                w = w[:-3] + 'y'
            elif w.endswith('sses') and len(w) > 5:
                w = w[:-2]
            elif w.endswith('ying') and len(w) > 5:
                w = w[:-4] + 'y'
            elif w.endswith('ing') and len(w) > 4:
                w = w[:-3]
            elif w.endswith('ed') and len(w) > 3:
                w = w[:-2]
            elif w.endswith('ly') and len(w) > 3:
                w = w[:-2]
            elif w.endswith('s') and len(w) > 3:
                w = w[:-1]
            out.append(w)
        return out
    return tokens


# ============================================================================
# Qwen helpers
# ============================================================================
def _qwen_generate(model, tokenizer, prompt, max_new_tokens=512):
    """Single Qwen generation call (greedy decoding)."""
    import torch
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    return response


# ============================================================================
# Session-summary generation (LongMemEval)
# ============================================================================
def _build_session_summary_prompt(messages):
    """Build the per-session summary prompt for LongMemEval."""
    formatted = []
    for i, m in enumerate(messages):
        role = m.get('role', 'unknown')
        content = m.get('content', '').strip()
        if len(content) > 600:
            content = content[:300] + "...[truncated]..." + content[-300:]
        formatted.append(f"[{i}] {role}: {content}")

    formatted_text = '\n'.join(formatted)

    prompt = (
        "Summarize the following conversation session in 3-5 sentences. "
        "Preserve all specific entities (names, places, dates), numbers, "
        "and key information mentioned. Focus on what the user and "
        "assistant discussed.\n\n"
        "Conversation:\n"
        f"{formatted_text}\n\n"
        "Summary:"
    )
    return prompt


def load_or_generate_summaries(items, qwen_path, cache_path):
    """Generate / cache session summaries for every haystack session."""
    cache = {}
    if os.path.exists(cache_path):
        print(f"[INFO] Loading existing summary cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] {len(cache)} sessions already cached")

    # Collect (qid, sid) pairs whose summary is still missing.
    needed = []  # list of (qid, sid, messages)
    for item in items:
        qid = item['question_id']
        sid_list = item.get('haystack_session_ids', [])
        sessions = item.get('haystack_sessions', [])
        for sid_idx, sid in enumerate(sid_list):
            cache_key = f"{qid}||{sid}"
            if cache_key in cache:
                continue
            msgs = sessions[sid_idx] if sid_idx < len(sessions) else []
            needed.append((qid, sid, msgs))

    if not needed:
        print("[INFO] All session summaries cached!")
        return cache

    print(f"[INFO] {len(needed)} sessions need summaries")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path, device_map='auto', trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    print(f"[INFO] Qwen loaded on {next(model.parameters()).device}")

    total = len(needed)
    t_start = time.time()

    for idx, (qid, sid, msgs) in enumerate(needed):
        if not msgs:
            cache_key = f"{qid}||{sid}"
            cache[cache_key] = '[empty session]'
            continue

        prompt = _build_session_summary_prompt(msgs)
        # Long sessions can exceed Qwen's context length, truncate.
        if len(prompt) > 12000:
            prompt = prompt[:6000] + "\n\n...[truncated]...\n\n" + prompt[-6000:]

        try:
            summary = _qwen_generate(model, tokenizer, prompt, max_new_tokens=300)
            # Strip every line break and bullet prefix to a single paragraph.
            summary = summary.replace('\n', ' ').strip()
            summary = re.sub(r'^[\-\*•·]+', '', summary).strip()
        except Exception as e:
            print(f"  [WARN] Qwen failed for {qid}/{sid}: {e}", file=sys.stderr)
            summary = '[generation failed]'

        cache_key = f"{qid}||{sid}"
        cache[cache_key] = summary if summary else '[empty summary]'

        if (idx + 1) % 20 == 0 or idx + 1 == total:
            pct = 100 * (idx + 1) / total
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (total - idx - 1) if idx > 0 else 0
            print(f"  [Summary] {idx + 1}/{total} ({pct:.1f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        # Save every 50 sessions.
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    return cache


# ============================================================================
# Turn-topic generation (per-session batch)
# ============================================================================
def _build_turn_topic_prompt(messages, sess_id, batch_size=20):
    """
    Build the per-session turn-topic prompt. Every line in the expected
    output is `<turn_id> || <topic phrase>`.
    """
    formatted = []
    turn_ids = []

    for i, m in enumerate(messages):
        role = m.get('role', 'unknown')
        content = m.get('content', '').strip()
        if len(content) > 500:
            content = content[:250] + "...[truncated]..." + content[-250:]

        tid = f"{sess_id}_{i}"
        formatted.append(f"[{i}] {role}: {content}")
        turn_ids.append(tid)

    if not formatted:
        return [], []

    # If a session has more than batch_size turns, split it into multiple batches.
    if len(formatted) <= batch_size:
        prompts = [(_build_single_topic_prompt(formatted, turn_ids, sess_id, 0, 1), turn_ids)]
    else:
        num_batches = (len(formatted) + batch_size - 1) // batch_size
        prompts = []
        for bi in range(num_batches):
            start = bi * batch_size
            end = min((bi + 1) * batch_size, len(formatted))
            batch_formatted = formatted[start:end]
            batch_turn_ids = turn_ids[start:end]
            prompts.append((_build_single_topic_prompt(
                batch_formatted, batch_turn_ids, sess_id, bi, num_batches
            ), batch_turn_ids))

    return prompts


def _build_single_topic_prompt(formatted, turn_ids, sess_id, batch_idx, num_batches):
    """Helper to assemble a single topic prompt for one batch."""
    formatted_text = '\n'.join(formatted)
    expected_ids_str = ', '.join(turn_ids)

    batch_info = f" (batch {batch_idx + 1}/{num_batches})" if num_batches > 1 else ""

    prompt = (
        f"For each turn in this conversation session {sess_id}{batch_info}, write ONE brief "
        f"topic phrase (5-10 words) describing the main subject of that turn. Be specific - "
        f"include names, entities, and key information mentioned.\n\n"
        f"Output format (one per line, EXACTLY):\n"
        f"turn_id || topic phrase\n\n"
        f"Expected turn IDs: {expected_ids_str}\n\n"
        f"Conversation:\n{formatted_text}\n\n"
        f"Topics:"
    )
    return prompt


def _parse_turn_topics(raw_output, expected_tids):
    """
    Parse the Qwen output where every line is of the form
    `turn_id || topic phrase` or `turn_id: topic phrase`.
    Returns {tid: topic_phrase}.
    """
    topics = {}
    for line in raw_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        if '||' in line:
            parts = line.split('||', 1)
            tid = parts[0].strip()
            phrase = parts[1].strip() if len(parts) > 1 else ''
        elif ':' in line:
            # Match the `<sess>_<idx>` format used by the prompt.
            m = re.match(r'(\d+_\d+)\s*[:：]\s*(.*)', line)
            if m:
                tid = m.group(1)
                phrase = m.group(2).strip()
            else:
                continue
        else:
            continue

        if tid not in expected_tids:
            continue
        phrase = phrase.strip().lstrip('-•·').strip()
        if len(phrase) >= 3:
            topics[tid] = phrase

    return topics


def load_or_generate_turn_topics(items, qwen_path, cache_path):
    """
    Load / generate the per-turn topic cache.

    Cache key:   f"{qid}||{tid}"   (tid = f"{sess_idx}_{msg_idx}")
    Cache value: topic phrase (string)
    """
    cache = {}
    if os.path.exists(cache_path):
        print(f"[INFO] Loading existing turn-topic cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] {len(cache)} turns already cached")

    # Collect (qid, sid_idx, messages) whose turns are not yet in the cache.
    needed = []
    for item in items:
        qid = item['question_id']
        sid_list = item.get('haystack_session_ids', [])
        sessions = item.get('haystack_sessions', [])
        for sid_idx, sid in enumerate(sid_list):
            msgs = sessions[sid_idx] if sid_idx < len(sessions) else []
            for i, m in enumerate(msgs):
                tid = f"{sid_idx}_{i}"
                cache_key = f"{qid}||{tid}"
                if cache_key in cache:
                    continue
                if (qid, sid_idx, msgs) not in needed:
                    needed.append((qid, sid_idx, msgs))

    if not needed:
        print("[INFO] All turn topics cached!")
        return cache

    print(f"[INFO] {len(needed)} sessions need topic generation")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path, device_map='auto', trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    print(f"[INFO] Qwen loaded on {next(model.parameters()).device}")

    total = len(needed)
    t_start = time.time()

    for idx, (qid, sid_idx, msgs) in enumerate(needed):
        if not msgs:
            continue

        prompts_with_tids = _build_turn_topic_prompt(msgs, sid_idx)
        all_parsed = {}

        for prompt, expected_tids in prompts_with_tids:
            if not prompt or not expected_tids:
                continue
            if len(prompt) > 12000:
                prompt = prompt[:6000] + "\n\n...[truncated]...\n\n" + prompt[-6000:]

            mnt = max(300, len(expected_tids) * 50)
            try:
                raw = _qwen_generate(model, tokenizer, prompt, max_new_tokens=mnt)
                parsed = _parse_turn_topics(raw, set(expected_tids))
                all_parsed.update(parsed)
            except Exception as e:
                print(f"  [WARN] Qwen failed for {qid} session {sid_idx}: {e}", file=sys.stderr)

        # Fallback for turns that were not produced: first 80 chars of the original turn.
        for i, m in enumerate(msgs):
            tid = f"{sid_idx}_{i}"
            cache_key = f"{qid}||{tid}"
            if cache_key not in cache:
                if tid not in all_parsed:
                    orig = m.get('content', '').strip()[:80]
                    all_parsed[tid] = f"[auto] {orig}" if orig else "[auto] no topic"
                cache[cache_key] = all_parsed[tid]

        # Progress.
        if (idx + 1) % 20 == 0 or idx + 1 == total:
            pct = 100 * (idx + 1) / total
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (total - idx - 1) if idx > 0 else 0
            print(f"  [TurnTopic] {idx + 1}/{total} ({pct:.1f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    return cache


# ============================================================================
# Hierarchical graph construction per question
# ============================================================================
def build_graph_per_question(item, summary_cache, turn_topics_cache, embed_fn):
    """
    Build the hierarchical memory graph for one question.

    Returns
    -------
    graph = (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
             turn_all, turn_to_sess, turn_texts_ordered, bm25_turn,
             turns_dict)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    qid = item['question_id']
    sid_list = item.get('haystack_session_ids', [])
    sessions = item.get('haystack_sessions', [])

    sk_all = list(range(len(sid_list)))

    sessions_meta = {}      # sid -> {'text': session_summary}
    turns_dict = {}         # tid -> {'session': sid, 'role': ..., 'text': ...}
    turn_all = []
    turn_to_sess = {}

    # Sessions & turns.
    for sid_idx, sid in enumerate(sid_list):
        msgs = sessions[sid_idx] if sid_idx < len(sessions) else []
        cache_key = f"{qid}||{sid}"
        summary_text = summary_cache.get(cache_key, '').strip()
        if not summary_text:
            summary_text = '[no summary]'
        sessions_meta[sid_idx] = {'text': summary_text}

        for i, m in enumerate(msgs):
            tid = f"{sid_idx}_{i}"
            content = m.get('content', '').strip()
            turns_dict[tid] = {
                'session': sid_idx,
                'role': m.get('role', ''),
                'text': content
            }
            turn_all.append(tid)
            turn_to_sess[tid] = sid_idx

    # Session embeddings.
    sess_texts = [sessions_meta[s]['text'] for s in sk_all]
    sess_embs = embed_fn(sess_texts)

    # Turn embeddings.
    turn_texts_ordered = []
    for tid in turn_all:
        turn = turns_dict[tid]
        txt = turn['text']
        role = turn['role']
        if not txt:
            txt = ' '
        turn_texts_ordered.append(f"{role}: {txt}" if role else txt)
    turn_embs = embed_fn(turn_texts_ordered)

    # Turn BM25 (with light Porter-style stemming).
    turn_tokens = [tokenize(t, stem=True) for t in turn_texts_ordered]
    bm25_turn = BM25(turn_tokens)

    # Per-session topic clusters.
    topic_clusters = []
    cluster_idx = 0
    for sid_idx in sk_all:
        msgs = sessions[sid_idx] if sid_idx < len(sessions) else []
        turn_topics_in_session = {}
        for i, m in enumerate(msgs):
            tid = f"{sid_idx}_{i}"
            cache_key = f"{qid}||{tid}"
            topic_text = turn_topics_cache.get(cache_key, '').strip()
            if not topic_text:
                topic_text = (m.get('content', '')[:80]).strip()
            turn_topics_in_session[tid] = topic_text

        if not turn_topics_in_session:
            continue

        tids = list(turn_topics_in_session.keys())
        topic_texts = [turn_topics_in_session[tid] for tid in tids]

        topic_embs_raw = embed_fn(topic_texts)
        if topic_embs_raw.ndim == 1:
            topic_embs_raw = topic_embs_raw.reshape(1, -1)

        n_turns = len(tids)
        if n_turns >= 3:
            sim_matrix = cosine_similarity(topic_embs_raw)
            distance_matrix = 1.0 - sim_matrix
            np.fill_diagonal(distance_matrix, 0.0)
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.55,
                    metric='precomputed', linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
            except:
                labels = list(range(n_turns))
        elif n_turns == 2:
            sim = cosine_similarity(topic_embs_raw)[0, 1] if topic_embs_raw.shape[0] == 2 else 0.0
            labels = [0, 0] if sim > 0.55 else [0, 1]
        else:
            labels = [0]

        cluster_groups = defaultdict(list)
        for i, lab in enumerate(labels):
            cluster_groups[int(lab)].append((tids[i], topic_texts[i], topic_embs_raw[i]))

        for lab, members in cluster_groups.items():
            member_tids = [m[0] for m in members]
            member_texts = [m[1] for m in members]
            member_embs = np.array([m[2] for m in members])
            best_text = max(member_texts, key=len) if member_texts else ''

            topic_clusters.append({
                'id': f'topic_{cluster_idx}',
                'text': best_text,
                'turns': member_tids,
                'session': sid_idx,
                'embedding': member_embs.mean(axis=0) if len(member_embs) > 0
                             else np.zeros(384, dtype=np.float32)
            })
            cluster_idx += 1

    # Topic-cluster embeddings.
    topic_texts_all = [tc['text'] for tc in topic_clusters]
    if topic_texts_all:
        topic_embs = embed_fn(topic_texts_all)
        if topic_embs.ndim == 1:
            topic_embs = topic_embs.reshape(1, -1)
    else:
        topic_embs = np.zeros((0, 384), dtype=np.float32)

    topic_to_turns = {tci: tc['turns'] for tci, tc in enumerate(topic_clusters)}

    return (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
            turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict)


# ============================================================================
# Three-path hybrid recall + graph re-ranking
# ============================================================================
def _cosine_scores(query_emb, doc_embs):
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    return np.dot(d, q)


def _norm(scores_dict):
    if not scores_dict:
        return scores_dict
    vals = list(scores_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {k: 0.0 for k in scores_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores_dict.items()}


def graph_hybrid_recall_rerank(query, meta, weights, embed_fn,
                               sess_k=3, topic_k=5, bm25_n=100, top_k=20):
    """
    Three-path recall -> merge & de-duplicate -> multi-signal re-ranking.
    """
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict) = meta

    w_bm25, w_turn, w_topic = weights
    q_emb = embed_fn(query)
    q_toks = set(tokenize(query, stem=True))

    candidate_tids = set()

    # Path 1: BM25 top-N.
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    for i in bm25_topn:
        candidate_tids.add(turn_all[i])

    # Path 2: MiniLM -> top-K sessions -> all turns of the session.
    if len(sess_embs) > 0:
        s_scores = _cosine_scores(q_emb, sess_embs)
        sess_topk = np.argsort(s_scores)[::-1][:sess_k]
        for si in sess_topk:
            sk = sk_all[si]
            for i, tid in enumerate(turn_all):
                if turn_to_sess[tid] == sk:
                    candidate_tids.add(tid)

    # Path 3: MiniLM -> top-K topic clusters -> only turns inside the cluster.
    if len(topic_embs) > 0 and len(topic_to_turns) > 0:
        tc_scores = _cosine_scores(q_emb, topic_embs)
        tc_topk = np.argsort(tc_scores)[::-1][:topic_k]
        for tci in tc_topk:
            for tid in topic_to_turns.get(tci, []):
                candidate_tids.add(tid)

    if not candidate_tids:
        return []

    pool = list(candidate_tids)
    pool_indices = [turn_all.index(tid) for tid in pool]

    # BM25 score (min-max-normalised inside the pool).
    bm25_pool = {tid: float(bm25_scores_all[idx]) for tid, idx in zip(pool, pool_indices)}
    bm25_norm = _norm(bm25_pool)

    # Turn-level semantic score.
    cand_embs = turn_embs[pool_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    turn_pool = {pool[i]: float(t_cos[i]) for i in range(len(pool))}
    turn_norm = _norm(turn_pool)

    # Topic-aware semantic score (broadcast to the turns inside each cluster).
    turn_topic_score = {}
    if len(topic_embs) > 0:
        tc_scores_all = _cosine_scores(q_emb, topic_embs)
        for tci in range(len(topic_to_turns)):
            score = float(tc_scores_all[tci])
            for tid in topic_to_turns.get(tci, []):
                turn_topic_score[tid] = max(turn_topic_score.get(tid, -999), score)

    topic_pool = {tid: turn_topic_score.get(tid, 0.0) for tid in pool}
    topic_norm = _norm(topic_pool)

    # Weighted final score.
    final = {}
    for tid in pool:
        final[tid] = (w_bm25 * bm25_norm.get(tid, 0) +
                      w_turn * turn_norm.get(tid, 0) +
                      w_topic * topic_norm.get(tid, 0))

    ranked = sorted(final, key=final.get, reverse=True)[:top_k]
    return ranked


# ============================================================================
# Ablation: each path on its own
# ============================================================================
def ablation_bm25_only(query, meta, bm25_n, top_k):
    """Ablation: BM25 recall + BM25 re-rank."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict) = meta
    q_toks = set(tokenize(query, stem=True))
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    pool_tids = [turn_all[i] for i in bm25_topn]
    pool_scores = [float(bm25_scores_all[i]) for i in bm25_topn]
    ranked = sorted(zip(pool_tids, pool_scores), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


def ablation_sem_session_only(query, meta, embed_fn, sess_k, top_k):
    """Ablation: Session path + turn-level cosine."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict) = meta
    if len(sess_embs) == 0:
        return []
    q_emb = embed_fn(query)
    s_scores = _cosine_scores(q_emb, sess_embs)
    sess_topk = np.argsort(s_scores)[::-1][:sess_k]
    candidate_tids = []
    candidate_indices = []
    for si in sess_topk:
        sk = sk_all[si]
        for i, tid in enumerate(turn_all):
            if turn_to_sess[tid] == sk:
                candidate_tids.append(tid)
                candidate_indices.append(i)
    if not candidate_tids:
        return []
    cand_embs = turn_embs[candidate_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    ranked = sorted(zip(candidate_tids, t_cos), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


def ablation_sem_topic_only(query, meta, embed_fn, topic_k, top_k):
    """Ablation: Topic-cluster path + turn-level cosine."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict) = meta
    if len(topic_embs) == 0 or len(topic_to_turns) == 0:
        return []
    q_emb = embed_fn(query)
    tc_scores = _cosine_scores(q_emb, topic_embs)
    tc_topk = np.argsort(tc_scores)[::-1][:topic_k]
    candidate_tids = []
    seen = set()
    for tci in tc_topk:
        for tid in topic_to_turns.get(tci, []):
            if tid not in seen:
                candidate_tids.append(tid)
                seen.add(tid)
    if not candidate_tids:
        return []
    candidate_indices = [turn_all.index(tid) for tid in candidate_tids]
    cand_embs = turn_embs[candidate_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    ranked = sorted(zip(candidate_tids, t_cos), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


# ============================================================================
# BM25 implementation
# ============================================================================
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.lens = np.array([len(d) for d in docs])
        self.avgdl = np.mean(self.lens) if len(self.lens) > 0 else 1.0
        self.idf = {}
        self.postings = defaultdict(list)
        self.tf_cache = [defaultdict(int) for _ in docs]
        for did, doc in enumerate(docs):
            seen = set()
            for t in doc:
                self.tf_cache[did][t] += 1
                if t not in seen:
                    self.postings[t].append(did)
                    seen.add(t)
            for t in seen:
                self.idf[t] = self.idf.get(t, 0) + 1
        for t, df in self.idf.items():
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query_tokens, top_k=10):
        scores = self.score_docs(query_tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return list(idx)

    def score_docs(self, query_tokens):
        scores = np.zeros(self.N)
        for t in query_tokens:
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for did in self.postings[t]:
                tf = self.tf_cache[did][t]
                scores[did] += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lens[did] / self.avgdl))
        return scores


# ============================================================================
# MiniLM embedding wrapper
# ============================================================================
def load_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"[INFO] Loading MiniLM: {EMBED_MODEL}")
    model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"[INFO] MiniLM ready on {device}")

    def embed(texts, bs=32):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        out = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inp = tokenizer(batch, return_tensors='pt', padding=True,
                            truncation=True, max_length=512)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                h = model(**inp).last_hidden_state
            mask = inp['attention_mask'].unsqueeze(-1).to(h.device)
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(e.cpu().float().numpy())
        if not out:
            return np.zeros((0, 384), dtype=np.float32)
        r = np.vstack(out)
        return r[0] if len(r) == 1 else r
    return embed


# ============================================================================
# Turn-level baselines
# ============================================================================
def rrf_retrieve_bm25_minilm(query, turn_all, bm25_turn, q_emb, turn_embs, top_k=20, rrf_k=60):
    """BM25 + MiniLM -> Reciprocal Rank Fusion."""
    q_toks = tokenize(query, stem=True)
    b_idx = bm25_turn.search(q_toks, top_k=100)
    b_rank = {turn_all[i]: rank + 1 for rank, i in enumerate(b_idx)}
    m_idx = embed_retrieve(q_emb, turn_embs, top_k=100)
    m_rank = {turn_all[i]: rank + 1 for rank, i in enumerate(m_idx)}
    all_turns = set(b_rank) | set(m_rank)
    max_rank = max(len(b_rank), len(m_rank)) + 1
    fused = {}
    for tid in all_turns:
        r1 = b_rank.get(tid, max_rank)
        r2 = m_rank.get(tid, max_rank)
        fused[tid] = 1.0 / (rrf_k + r1) + 1.0 / (rrf_k + r2)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]


def embed_retrieve(query_emb, doc_embs, top_k=20):
    scores = _cosine_scores(query_emb, doc_embs)
    idx = np.argsort(scores)[::-1][:top_k]
    return list(idx)


# ============================================================================
# Evaluation utilities
# ============================================================================
def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / max(len(gold), 1)


def mrr_score(ranked, gold):
    g = set(gold)
    for i, x in enumerate(ranked, 1):
        if x in g:
            return 1.0 / i
    return 0.0


def compute_f1(prediction, ground_truth):
    """Token-level F1."""
    def norm(s):
        return str(s).lower().strip().rstrip('.')
    pred = norm(prediction)
    gt = norm(ground_truth)
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    tp = len(pred_tokens & gt_tokens)
    if tp == 0:
        return 0.0
    p = tp / len(pred_tokens)
    r = tp / len(gt_tokens)
    return 2 * p * r / (p + r)


def compute_exact_match(prediction, ground_truth):
    def norm(s):
        return str(s).lower().strip().rstrip('.')
    return 1.0 if norm(prediction) == norm(ground_truth) else 0.0


def generate_answer_e2e(model, tokenizer, query, ranked_turns, turns_dict,
                        question_date=None, max_retrieved=20):
    """
    Use the top-N retrieved turns as context and have Qwen generate the answer.
    """
    top_turns = ranked_turns[:max_retrieved]
    turn_texts = []
    for tid in top_turns:
        turn = turns_dict.get(tid, {})
        role = turn.get('role', '')
        text = turn.get('text', '')
        if not text:
            text = ' '
        turn_texts.append(f"{role}: {text}" if role else text)

    context = '\n'.join(turn_texts)

    if len(context) > 6000:
        context = context[:3000] + '\n...[truncated]...\n' + context[-3000:]

    date_hint = ''
    if question_date:
        date_hint = f"The question was asked on {question_date}.\n"

    prompt = (
        "Based on the following conversation history, answer the question. "
        "Answer as concisely as possible - just give the direct answer, no explanation.\n\n"
        f"{date_hint}"
        "Conversation history:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    t0 = time.time()
    raw = _qwen_generate(model, tokenizer, prompt, max_new_tokens=200)
    gen_time = time.time() - t0

    answer = raw.split('\n')[0]
    return answer, gen_time


# ============================================================================
# Gold evidence collection
# ============================================================================
def collect_gold_evidence(item):
    """
    Collect the gold evidence turn IDs.

    Turn ID format: `{sid_idx}_{msg_idx}`.

    Strategy:
      - Primary: every message with `has_answer == true` inside the haystack
        session that contains the answer.
      - Fallback: if no `has_answer` flag is present, use the messages
        whose `session_id` is in `answer_session_ids`.
    """
    gold_tids = []
    sid_list = item.get('haystack_session_ids', [])
    sessions = item.get('haystack_sessions', [])
    answer_session_ids = item.get('answer_session_ids', [])

    for sid_idx, sid in enumerate(sid_list):
        msgs = sessions[sid_idx] if sid_idx < len(sessions) else []
        for i, m in enumerate(msgs):
            tid = f"{sid_idx}_{i}"
            has_answer = m.get('has_answer', None)
            if has_answer is None:
                # Fallback: treat every message from `answer_session_ids` as relevant.
                if sid in answer_session_ids:
                    gold_tids.append(tid)
            elif has_answer is True:
                gold_tids.append(tid)

    return gold_tids


# ============================================================================
# Main
# ============================================================================
QUESTION_TYPE_MAP = {
    'single-session-user': 'single-session-user',
    'single-session-assistant': 'single-session-assistant',
    'single-session-preference': 'single-session-preference',
    'knowledge-update': 'knowledge-update',
    'temporal-reasoning': 'temporal-reasoning',
    'multi-session': 'multi-session'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='longmemeval_s_cleaned.json')
    parser.add_argument('--qwen_path', type=str, default=None,
                        help='Path to Qwen2.5-7B-Instruct (only needed for the first run that generates summaries + turn topics).')
    parser.add_argument('--summary_cache', type=str,
                        default='longmemeval_summaries_v3.json')
    parser.add_argument('--turn_topic_cache', type=str,
                        default='turn_topics_cache_v3.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_qa', type=int, default=None,
                        help='Optional cap on the number of evaluated questions.')
    parser.add_argument('--weight_search', action='store_true')
    parser.add_argument('--grid_step', type=float, default=0.10)
    parser.add_argument('--sem_session_k', type=int, default=3)
    parser.add_argument('--sem_topic_k', type=int, default=5)
    parser.add_argument('--sem_bm25_n', type=int, default=100)
    parser.add_argument('--end_to_end', action='store_true',
                        help='Run end-to-end answer generation + F1/EM (requires --qwen_path).')
    parser.add_argument('--max_retrieved', type=int, default=20,
                        help='Number of retrieved turns fed into the E2E generator (default: 20).')
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    TOP_K = args.top_k

    # ===== Load data =====
    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    if args.max_qa:
        data = data[:args.max_qa]
        print(f"[INFO] {len(data)} questions (sampled).")

    # ===== Phase 0: load / generate caches =====
    if args.qwen_path:
        summary_cache = load_or_generate_summaries(
            data, args.qwen_path, args.summary_cache
        )
        turn_topics_cache = load_or_generate_turn_topics(
            data, args.qwen_path, args.turn_topic_cache
        )
    else:
        # Without Qwen we can only run if both caches already exist.
        if not os.path.exists(args.summary_cache):
            print(f"[ERROR] Summary cache not found: {args.summary_cache}")
            print("  First run: python graph_vs_baselines_LongMemEval.py "
                  "--qwen_path /path/to/Qwen2.5-7B-Instruct")
            return
        if not os.path.exists(args.turn_topic_cache):
            print(f"[ERROR] Turn topic cache not found: {args.turn_topic_cache}")
            return
        print(f"[INFO] Loading summary cache: {args.summary_cache}")
        with open(args.summary_cache, 'r', encoding='utf-8') as f:
            summary_cache = json.load(f)
        print(f"[INFO] Loading turn-topic cache: {args.turn_topic_cache}")
        with open(args.turn_topic_cache, 'r', encoding='utf-8') as f:
            turn_topics_cache = json.load(f)

    # ===== Load MiniLM =====
    embed = load_embedder()

    # ===== Phase 1: build graphs =====
    all_samples = []
    graph_cache = {}  # qid -> graph tuple (for re-use in the grid search).
    build_t0 = time.time()

    for qi, item in enumerate(data):
        print(f"[Q {qi + 1}/{len(data)}] {item['question_id']}  building graph + embedding...")
        meta = build_graph_per_question(item, summary_cache, turn_topics_cache, embed)
        (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
         turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict) = meta
        graph_cache[item['question_id']] = meta

        # Gold evidence (only from the haystack session ids).
        gold = collect_gold_evidence(item)
        if not gold:
            continue

        query = item.get('question', '')
        qtype = item.get('question_type', 'unknown')
        ans = item.get('answer', '')

        all_samples.append((item['question_id'], query, gold, qtype, turns_dict, ans))

    total_qa = len(all_samples)
    build_time = time.time() - build_t0
    print(f"\n[INFO] Collected {total_qa} questions with valid gold (build time: {build_time:.1f}s)")

    if total_qa == 0:
        print("[ERROR] No valid questions! Exiting.")
        return

    # ===== Phase 2: grid-search =====
    weights = (0.50, 0.40, 0.10)  # Default: w_bm25, w_turn, w_topic.
    retrieve_fn = lambda q, m, w: graph_hybrid_recall_rerank(
        q, m, w, embed, args.sem_session_k, args.sem_topic_k, args.sem_bm25_n, TOP_K
    )
    mode_name = 'Graph_HybridRecall'
    grid_results = None

    if args.weight_search:
        weight_grid = []
        n = int(1.0 / args.grid_step)
        values = [round(i * args.grid_step, 10) for i in range(n + 1)]
        for w1 in values:
            for w2 in values:
                w3 = 1.0 - w1 - w2
                if w3 < -0.0001 or w3 > 1.0001:
                    continue
                if any(abs(w3 - v) < 0.0001 for v in values):
                    weight_grid.append((w1, w2, round(w3, 10)))
        weight_grid = sorted(set(weight_grid))

        print(f"{'='*80}")
        print(f"WEIGHT GRID SEARCH: {len(weight_grid)} combinations (step={args.grid_step})")
        print(f"{'='*80}")

        all_grid_results = []
        best = {'weights': None, 'r10': 0.0}
        report_every = max(1, len(weight_grid) // 20)

        for gi, w in enumerate(weight_grid):
            r1_vals, r3_vals, r5_vals, r10_vals, mrr_vals = [], [], [], [], []
            for qid, query, gold, qtype, _turns, _ans in all_samples:
                meta = graph_cache[qid]
                ranked = retrieve_fn(query, meta, w)
                r1_vals.append(recall_at_k(ranked, gold, 1))
                r3_vals.append(recall_at_k(ranked, gold, 3))
                r5_vals.append(recall_at_k(ranked, gold, 5))
                r10_vals.append(recall_at_k(ranked, gold, 10))
                mrr_vals.append(mrr_score(ranked, gold))

            avg_r1 = float(np.mean(r1_vals))
            avg_r3 = float(np.mean(r3_vals))
            avg_r5 = float(np.mean(r5_vals))
            avg_r10 = float(np.mean(r10_vals))
            avg_mrr = float(np.mean(mrr_vals))

            label_names = ['bm25', 'turn', 'topic']
            combo_result = {
                'weights': {n: round(v, 2) for n, v in zip(label_names, w)},
                'R@1': avg_r1, 'R@3': avg_r3, 'R@5': avg_r5,
                'R@10': avg_r10, 'MRR': avg_mrr
            }
            all_grid_results.append(combo_result)

            is_new_best = avg_r10 > best['r10']
            if is_new_best:
                best = {'weights': w, 'r1': avg_r1, 'r3': avg_r3, 'r5': avg_r5,
                        'r10': avg_r10, 'mrr': avg_mrr}

            if gi % report_every == 0 or gi == len(weight_grid) - 1:
                pct = 100 * (gi + 1) / len(weight_grid)
                marker = ' *' if is_new_best else '  '
                print(f"  [{gi+1:>4}/{len(weight_grid)} {pct:5.1f}%]{marker} "
                      f"R@10={avg_r10:.4f} (best={best['r10']:.4f})")

        weights = best['weights']
        all_grid_results.sort(key=lambda x: x['R@10'], reverse=True)
        grid_results = all_grid_results

        print(f"\n[Grid Search] BEST -> R@10={best['r10']:.4f}  R@5={best['r5']:.4f}  "
              f"MRR={best['mrr']:.4f}")
        label_names = ['bm25', 'turn', 'topic']
        print(f"  Weights: {dict(zip(label_names, [f'{v:.2f}' for v in weights]))}")
        print(f"\n[Grid Search] Top 10 of {len(all_grid_results)} combinations:")
        for rank, r in enumerate(all_grid_results[:10], 1):
            ws = r['weights']
            print(f"  #{rank:<3} R@10={r['R@10']:.4f} R@5={r['R@5']:.4f} MRR={r['MRR']:.4f}  "
                  f"bm25={ws['bm25']:.2f} turn={ws['turn']:.2f} topic={ws['topic']:.2f}")
    else:
        print(f"[INFO] Default weights: bm25={weights[0]:.2f} "
              f"turn={weights[1]:.2f} topic={weights[2]:.2f}")

    # ===== Phase 3: retrieval evaluation =====
    methods = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
               'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    overall = {m: defaultdict(list) for m in methods}
    per_qtype = {m: defaultdict(lambda: defaultdict(list)) for m in methods}
    retrieval_times = defaultdict(list)

    all_ranked = []  # Cache per-question results for the E2E phase.

    for eval_idx, (qid, query, gold, qtype, turns_dict, ans) in enumerate(all_samples):
        meta = graph_cache[qid]
        (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
         turn_all, turn_to_sess, turn_texts_ordered, bm25_turn, turns_dict_meta) = meta

        # Main: three-path recall + re-ranking.
        t0 = time.time()
        g_ranked = retrieve_fn(query, meta, weights)
        retrieval_times[mode_name].append(time.time() - t0)

        q_emb = embed(query)
        q_toks = set(tokenize(query, stem=True))

        # BM25 + MiniLM RRF.
        t0 = time.time()
        rrf_ranked = rrf_retrieve_bm25_minilm(query, turn_all, bm25_turn, q_emb, turn_embs, top_k=TOP_K, rrf_k=60)
        retrieval_times['BM25+MiniLM_RRF'].append(time.time() - t0)

        # MiniLM (turn-level).
        t0 = time.time()
        m_idx = embed_retrieve(q_emb, turn_embs, top_k=TOP_K)
        m_ranked = [turn_all[i] for i in m_idx]
        retrieval_times['MiniLM'].append(time.time() - t0)

        # BM25.
        t0 = time.time()
        b_idx = bm25_turn.search(q_toks, top_k=TOP_K)
        b_ranked = [turn_all[i] for i in b_idx]
        retrieval_times['BM25'].append(time.time() - t0)

        # Ablation: each path on its own.
        t0 = time.time()
        abl_b = ablation_bm25_only(query, meta, args.sem_bm25_n, TOP_K)
        retrieval_times['Abl_BM25'].append(time.time() - t0)
        t0 = time.time()
        abl_s = ablation_sem_session_only(query, meta, embed, args.sem_session_k, TOP_K)
        retrieval_times['Abl_SemSession'].append(time.time() - t0)
        t0 = time.time()
        abl_t = ablation_sem_topic_only(query, meta, embed, args.sem_topic_k, TOP_K)
        retrieval_times['Abl_SemTopic'].append(time.time() - t0)

        all_ranked.append({
            'qid': qid, 'query': query, 'gold': gold, 'qtype': qtype,
            'turns_dict': turns_dict, 'answer': ans,
            'question_date': next((it.get('question_date', '') for it in data
                                   if it.get('question_id') == qid), ''),
            mode_name: g_ranked,
            'BM25+MiniLM_RRF': rrf_ranked,
            'MiniLM': m_ranked,
            'BM25': b_ranked,
            'Abl_BM25': abl_b,
            'Abl_SemSession': abl_s,
            'Abl_SemTopic': abl_t,
        })

        for method, ranked in [(mode_name, g_ranked), ('BM25+MiniLM_RRF', rrf_ranked),
                                ('MiniLM', m_ranked), ('BM25', b_ranked),
                                ('Abl_BM25', abl_b), ('Abl_SemSession', abl_s),
                                ('Abl_SemTopic', abl_t)]:
            for k in [1, 3, 5, 10]:
                overall[method][f'R@{k}'].append(recall_at_k(ranked, gold, k))
                per_qtype[method][qtype][f'R@{k}'].append(recall_at_k(ranked, gold, k))
            overall[method]['MRR'].append(mrr_score(ranked, gold))
            per_qtype[method][qtype]['MRR'].append(mrr_score(ranked, gold))

        # Progress.
        if (eval_idx + 1) % 50 == 0 or eval_idx + 1 == total_qa:
            pct = 100 * (eval_idx + 1) / total_qa
            print(f"  [Retrieval Eval] {eval_idx + 1}/{total_qa} ({pct:.1f}%)", flush=True)

    # ===== Phase 4: end-to-end evaluation =====
    baseline_e2e_methods = ['BM25+MiniLM_RRF', 'MiniLM', 'BM25']
    e2e_methods = [mode_name] + baseline_e2e_methods + ['Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    e2e_all = {}

    if args.end_to_end and args.qwen_path:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n{'='*80}")
        print("END-TO-END ANSWER GENERATION + F1/EM EVALUATION")
        print(f"{'='*80}")

        e2e_tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
        if e2e_tokenizer.pad_token is None:
            e2e_tokenizer.pad_token = e2e_tokenizer.eos_token
        e2e_model = AutoModelForCausalLM.from_pretrained(
            args.qwen_path, device_map='auto', trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        e2e_model.eval()
        print(f"[INFO] E2E Qwen loaded on {next(e2e_model.parameters()).device}")

        e2e_f1 = {m: [] for m in e2e_methods}
        e2e_em = {m: [] for m in e2e_methods}
        e2e_gen_time = {m: [] for m in e2e_methods}
        e2e_total_time = {m: [] for m in e2e_methods}

        e2e_total = len(all_ranked)
        for e2e_idx, r in enumerate(all_ranked):
            query = r['query']
            turns_dict = r['turns_dict']
            ans = r['answer']
            if not ans:
                continue

            for em_name in e2e_methods:
                ranked = r[em_name]
                pred_answer, gen_t = generate_answer_e2e(
                    e2e_model, e2e_tokenizer, query, ranked, turns_dict,
                    question_date=r.get('question_date', ''),
                    max_retrieved=args.max_retrieved
                )
                ret_t = retrieval_times[em_name][e2e_idx]
                f1 = compute_f1(pred_answer, ans)
                em = compute_exact_match(pred_answer, ans)
                e2e_f1[em_name].append(f1)
                e2e_em[em_name].append(em)
                e2e_gen_time[em_name].append(gen_t)
                e2e_total_time[em_name].append(ret_t + gen_t)

            if (e2e_idx + 1) % 20 == 0 or e2e_idx + 1 == e2e_total:
                pct = 100 * (e2e_idx + 1) / e2e_total
                parts = [f"{em_name}:F1={np.mean(e2e_f1[em_name]):.4f}"
                         for em_name in e2e_methods]
                print(f"  [E2E] {e2e_idx + 1}/{e2e_total} ({pct:.1f}%) | " + " | ".join(parts),
                      flush=True)

        for em_name in e2e_methods:
            e2e_all[em_name] = {
                'f1': float(np.mean(e2e_f1[em_name])) if e2e_f1[em_name] else 0.0,
                'exact_match': float(np.mean(e2e_em[em_name])) if e2e_em[em_name] else 0.0,
                'avg_generation_time_s': float(np.mean(e2e_gen_time[em_name]))
                                        if e2e_gen_time[em_name] else 0.0,
                'avg_total_pipeline_time_s': float(np.mean(e2e_total_time[em_name]))
                                             if e2e_total_time[em_name] else 0.0,
                'num_evaluated': len(e2e_f1[em_name])
            }

        print("\n[E2E Results]")
        for em_name in e2e_methods:
            avg_f1 = float(np.mean(e2e_f1[em_name])) if e2e_f1[em_name] else 0.0
            avg_em = float(np.mean(e2e_em[em_name])) if e2e_em[em_name] else 0.0
            avg_gt = float(np.mean(e2e_gen_time[em_name])) if e2e_gen_time[em_name] else 0.0
            avg_total = float(np.mean(e2e_total_time[em_name])) if e2e_total_time[em_name] else 0.0
            print(f"  {em_name}: F1={avg_f1:.4f}  EM={avg_em:.4f}  "
                  f"ret+gen={avg_total:.2f}s  gen_only={avg_gt:.2f}s")

        del e2e_model
        torch.cuda.empty_cache()

    # ===== Phase 5: print and save =====
    print("\n" + "=" * 90)
    suffix = " (grid-best)" if args.weight_search else ""
    all_display_methods = [f"{mode_name}{suffix}", 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
                           'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    print(f"RESULTS ({total_qa} questions, top_k={TOP_K}) [MRR@{TOP_K}]")
    print("=" * 90)
    metrics = ['R@1', 'R@3', 'R@5', 'R@10', 'MRR']
    header = f"{'Metric':<8}" + "".join(f"{m:>22}" for m in all_display_methods)
    print(header)
    print("-" * 90)
    for met in metrics:
        vals = [np.mean(overall[m][met]) for m in methods]
        print(f"{met:<8}" + "".join(f"{v:>22.4f}" for v in vals))

    if e2e_all:
        print("\n" + "-" * 90)
        print("END-TO-END (F1 / EM / Time)")
        e2e_display = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
                       'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
        header2 = f"{'Metric':<14}" + "".join(f"{m:>22}" for m in e2e_display)
        print(header2)
        print("-" * 90)
        for met_name, met_key in [('F1', 'f1'), ('EM', 'exact_match')]:
            row = f"{met_name:<14}"
            for em_name in e2e_display:
                val = e2e_all.get(em_name, {}).get(met_key, 0.0)
                row += f"{val:>22.4f}"
            print(row)
        row = f"{'ret+gen(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_total_pipeline_time_s', 0.0)
            row += f"{val:>22.2f}"
        print(row)
        row = f"{'gen_only(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_generation_time_s', 0.0)
            row += f"{val:>22.2f}"
        print(row)

    print("\n" + "-" * 90)
    print("AVERAGE RETRIEVAL TIME PER QUERY (ms)")
    print(f"{'Method':<22}" + f"{'Avg Time':>10}")
    print("-" * 35)
    for m in methods:
        times = retrieval_times.get(m, [])
        avg_ms = np.mean(times) * 1000 if times else 0
        print(f"{m:<22} {avg_ms:>9.1f}ms")

    print("\n" + "-" * 90)
    print("PER-QUESTION-TYPE (R@5)")
    print(f"{'Question Type':<28}" + "".join(f"{m:>22}" for m in all_display_methods))
    print("-" * 90)
    all_qtypes = sorted(set(qt for _, _, _, qt, _, _ in all_samples))
    for qt in all_qtypes:
        row = f"{qt:<28}"
        for m in methods:
            vals = per_qtype[m][qt].get('R@5', [])
            row += f"{np.mean(vals):>22.4f}" if vals else f"{'N/A':>22}"
        print(row)

    # Save to JSON.
    out = {
        'version': 'graph_hybrid_recall_v3_update',
        'method': (f'BM25(top{args.sem_bm25_n}) + MiniLM->sess(top{args.sem_session_k}) '
                   f'+ MiniLM->topic_cluster(top{args.sem_topic_k})'),
        'top_k': TOP_K,
        'sem_session_k': args.sem_session_k,
        'sem_topic_k': args.sem_topic_k,
        'sem_bm25_n': args.sem_bm25_n,
        'total_qa': total_qa,
        'weight_search': args.weight_search,
        'weights': {k: float(v) for k, v in zip(['bm25', 'turn', 'topic'], weights)},
        'pre_build_time_s': round(build_time, 1),
        'avg_retrieval_time_ms': {},
        'overall': {
            m: {k: float(np.mean(v)) if v else 0.0 for k, v in ov.items()}
            for m, ov in overall.items()
        },
        'ablation': {
            'Abl_BM25': 'Path 1 only: BM25 query->turn recall.',
            'Abl_SemSession': 'Path 2 only: MiniLM(query, session_summary) recall.',
            'Abl_SemTopic': 'Path 3 only: MiniLM(query, topic_cluster) recall.',
        },
        'per_qtype_r5': {},
        'per_qtype_r10': {}
    }
    for m in methods:
        times = retrieval_times.get(m, [])
        out['avg_retrieval_time_ms'][m] = float(np.mean(times) * 1000) if times else 0.0

    for qt in all_qtypes:
        out['per_qtype_r5'][qt] = {}
        out['per_qtype_r10'][qt] = {}
        for m in methods:
            vals5 = per_qtype[m][qt].get('R@5', [])
            vals10 = per_qtype[m][qt].get('R@10', [])
            out['per_qtype_r5'][qt][m] = float(np.mean(vals5)) if vals5 else None
            out['per_qtype_r10'][qt][m] = float(np.mean(vals10)) if vals10 else None

    if e2e_all:
        out['e2e_all'] = e2e_all

    if grid_results is not None:
        out['grid_search'] = {
            'step': args.grid_step,
            'num_combos_tested': len(grid_results),
            'all_results': grid_results
        }

    outpath = 'graph_hybrid_recall_v3_update_result.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()