"""
LongMemEval → three-way hybrid recall + graph-signal reranking (v3 update)
================================================================================
Data: longmemeval_s_cleaned.json (500 items, each item = 1 QA + N haystack sessions)

Improvements:
  1. BM25 normalization: added Porter Stemmer stemming
  2. Event-level → Turn-Topic level: Qwen-7B generates a topic summary for each turn,
     and turns with similar topics within the same session are merged into a topic cluster
  3. End-to-end evaluation: Recall@10 retrieval → Qwen answer generation → F1/EM
  4. Timing: average per-question cost of graph construction / retrieval / generation

Three-way recall:
  Path 1: BM25(query, turn)              top-100 → keyword backbone
  Path 2: MiniLM(query, session_summary)  top-3   → all turns (semantic → coarse topic)
  Path 3: MiniLM(query, topic_cluster)    top-5   → cluster turns (semantic → fine-grained topic)
Merge and deduplicate → three-way layered score reranking (w_bm25 + w_turn + w_topic) → top-20

Gold: the turn ID of the message with has_answer=true

Baselines (turn level):
  - BM25 (turn-level)
  - MiniLM (turn-level cosine)
  - BM25+MiniLM RRF (turn-level)

Usage:
  First run (topic generation requires Qwen):
    python graph_vs_baselines_v3_copy_update.py --qwen_path /path/to/Qwen2.5-7B-Instruct --weight_search
  Later runs (loaded from cache):
    python graph_vs_baselines_v3_copy_update.py --weight_search
  End-to-end evaluation:
    python graph_vs_baselines_v3_copy_update.py --qwen_path /path/to/Qwen2.5-7B-Instruct --end_to_end
"""

import argparse
import json, sys, io, math, os, time, re
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


# ═══════════════════════════════════════════
# Tokenize (with Porter Stemmer)
# ═══════════════════════════════════════════
_STOPS = {'the','a','an','is','are','was','were','in','on','at','to','for',
          'of','with','and','or','by','from','it','its','i','you','he','she',
          'we','they','my','your','his','her','our','their','me','him','us',
          'them','that','this','these','those','be','been','being','have','has',
          'had','do','does','did','will','would','can','could','should','may',
          'might','not','no','but','if','so','very','just','about','also',
          'what','when','where','who','how','why','which','whom'}

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    _HAS_STEMMER = True
except ImportError:
    _stemmer = None
    _HAS_STEMMER = False


def tokenize(text):
    t = text.lower()
    for ch in '?!.,;:\"\'()[]{}-\n\r':
        t = t.replace(ch, ' ')
    words = [w for w in t.split() if w not in _STOPS and len(w) > 1]
    if _HAS_STEMMER and _stemmer:
        words = [_stemmer.stem(w) for w in words]
    return words


# ═══════════════════════════════════════════
# Qwen helper functions
# ═══════════════════════════════════════════
def _truncate_session(messages, max_chars=6000):
    """Truncate an overly long session."""
    if not messages:
        return ""
    truncated = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        if len(content) > 3000:
            content = content[:1500] + "\n...[truncated]...\n" + content[-1500:]
        truncated.append(f"[{role}]: {content}")
    text = "\n\n".join(truncated)
    if len(text) > max_chars:
        head = "\n\n".join(truncated[:max(1, len(truncated)//3)])
        tail = "\n\n".join(truncated[-max(1, len(truncated)//3):])
        text = head + "\n\n...[middle messages omitted]...\n\n" + tail
    return text


def _qwen_generate(model, tokenizer, prompt, max_new_tokens=256):
    """One Qwen generation call."""
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


def _load_qwen_model(qwen_path):
    """Load the Qwen model and tokenizer."""
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
    return model, tokenizer


# ═══════════════════════════════════════════
# Session summary generation (unchanged)
# ═══════════════════════════════════════════
def load_or_generate_summaries(data, qwen_path, cache_path):
    """Load cached summaries, or generate them item by item and persist each one
    immediately."""
    import torch

    summaries = {}
    if os.path.exists(cache_path):
        print(f"[INFO] Loading existing summary cache from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
        print(f"[INFO] {len(summaries)} sessions already cached")

    all_needed = {}
    for item in data:
        for i, sid in enumerate(item['haystack_session_ids']):
            if sid not in summaries and sid not in all_needed:
                all_needed[sid] = item['haystack_sessions'][i]

    if not all_needed:
        print("[INFO] All sessions already cached!")
        return summaries

    print(f"[INFO] {len(all_needed)} sessions need session summary (Qwen: {qwen_path})")
    model, tokenizer = _load_qwen_model(qwen_path)

    needed_list = list(all_needed.items())
    total_needed = len(needed_list)
    batch_size = 4
    t_start = time.time()
    save_every = 20

    for start in range(0, total_needed, batch_size):
        batch = needed_list[start:start + batch_size]
        for sid, messages in batch:
            text = _truncate_session(messages)

            # Event Summary
            event_prompt = (
                "Extract every distinct event from this conversation. "
                "For each event, write ONE sentence that MUST include: "
                "what happened, WHO was involved (names/roles), and WHEN (dates/times if available). "
                "Keep specific entities (names, products, numbers, locations) exactly as they appear - do NOT generalize them. "
                "List events in chronological order. "
                "If no events are described, extract the main discussion topics as events instead.\n\n"
                f"Conversation:\n{text}\n\nEvents:"
            )
            events_raw = _qwen_generate(model, tokenizer, event_prompt, max_new_tokens=400)
            events = []
            for line in events_raw.split('\n'):
                line = line.strip().lstrip('-•·1234567890. ').strip()
                if len(line) > 10 and line.lower() != 'no specific events.':
                    events.append(line)

            # Session Summary
            summary_prompt = (
                "Summarize this conversation in 3-5 sentences. "
                "CRITICAL: Preserve ALL specific entities exactly as they appear - "
                "names of people, products, tools, brands, locations, dates, prices, numbers. "
                "Do NOT replace them with general categories. "
                "Include every topic discussed, every recommendation made, and every decision reached. "
                "Write in past tense, third person.\n\n"
                f"Conversation:\n{text}\n\nSummary:"
            )
            session_summary = _qwen_generate(model, tokenizer, summary_prompt, max_new_tokens=250)

            summaries[sid] = {
                'session_summary': session_summary,
                'events': events[:10]
            }

        done = start + len(batch)
        pct = 100 * done / total_needed
        elapsed = time.time() - t_start
        eta = elapsed / done * (total_needed - done)
        print(f"  [Summary Gen] {done}/{total_needed} ({pct:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", file=sys.stderr)

        if done % save_every == 0 or done >= total_needed:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, ensure_ascii=False)

        if done % 100 == 0:
            torch.cuda.empty_cache()

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"[INFO] {total_needed} summaries generated in {total_time/60:.1f} min, cached to {cache_path}")

    del model
    torch.cuda.empty_cache()
    return summaries


# ═══════════════════════════════════════════
# Turn topic generation (new: Qwen per-session batch)
# ═══════════════════════════════════════════
def _parse_turn_topics_v3(raw_output, expected_tids):
    """Parse the Qwen output: 'turn_id || topic phrase' → {tid: phrase}"""
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
            m = re.match(r'(\d+_\d+|turn_\d+)\s*[:：]\s*(.*)', line)
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


def load_or_generate_turn_topics_v3(data, qwen_path, cache_path, summaries):
    """
    LongMemEval-specific: generate a topic summary for every turn.
    Cache key: f"{session_id}||{item_idx}_{msg_idx}"
    """
    cache = {}
    if os.path.exists(cache_path):
        print(f"[INFO] Loading existing turn-topic cache from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] {len(cache)} turns already cached")

    # Collect all the turns that are needed
    all_needed = {}  # composite_key → (item_idx, session_id, session_idx, messages)
    for item_idx, item in enumerate(data):
        for si, sid in enumerate(item['haystack_session_ids']):
            msgs = item['haystack_sessions'][si]
            has_missing = False
            for mi, m in enumerate(msgs):
                tid = f"{si}_{mi}"
                ck = f"{sid}||{tid}"
                if ck not in cache:
                    has_missing = True
            if has_missing:
                composite = f"{item_idx}||{sid}"
                all_needed[composite] = (item_idx, sid, si, msgs)

    if not all_needed:
        print("[INFO] All turn topics already cached!")
        return cache

    total_sessions = len(all_needed)
    total_turns = sum(len(v[3]) for v in all_needed.values())
    print(f"[INFO] {total_sessions} sessions ({total_turns} turns) need topic generation")

    import torch
    model, tokenizer = _load_qwen_model(qwen_path)

    needed_list = list(all_needed.values())
    t_start = time.time()
    save_every = 10
    BATCH_SIZE = 20  # at most 20 turns per batch

    for idx, (item_idx, sid, si, messages) in enumerate(needed_list):
        session_text = _truncate_session(messages)

        # Build every turn ID
        all_tids = [f"{si}_{mi}" for mi in range(len(messages))]

        # Process in batches
        all_parsed = {}
        num_batches = (len(messages) + BATCH_SIZE - 1) // BATCH_SIZE

        for bi in range(num_batches):
            b_start = bi * BATCH_SIZE
            b_end = min((bi + 1) * BATCH_SIZE, len(messages))
            batch_msgs = messages[b_start:b_end]
            batch_tids = all_tids[b_start:b_end]

            # Format this batch of turns
            batch_lines = []
            for m, tid in zip(batch_msgs, batch_tids):
                role = m.get('role', 'user')
                content = m.get('content', '').strip()
                if len(content) > 400:
                    content = content[:200] + "...[truncated]..." + content[-200:]
                batch_lines.append(f"{tid} [{role}]: {content}")

            batch_label = f"sid={sid}[b{bi+1}/{num_batches}]" if num_batches > 1 else f"sid={sid}"
            prompt = (
                f"For each turn in this conversation, write ONE brief topic phrase (5-10 words) "
                f"describing the main subject. Be specific - include names, entities, and key information.\n\n"
                f"Output format EXACTLY (one per line):\n"
                f"turn_id || topic phrase\n\n"
                f"Conversation:\n{chr(10).join(batch_lines)}\n\nTopics:"
            )

            if len(prompt) > 8000:
                prompt = prompt[:4000] + "\n...[truncated]...\n" + prompt[-4000:]

            try:
                mnt = max(300, len(batch_tids) * 44)
                raw = _qwen_generate(model, tokenizer, prompt, max_new_tokens=mnt)
                parsed = _parse_turn_topics_v3(raw, set(batch_tids))
                if num_batches > 1:
                    print(f"    [batch {bi+1}/{num_batches}] Qwen={len(parsed)}/{len(batch_tids)}",
                          file=sys.stderr)
            except Exception as e:
                print(f"  [WARN] Qwen failed for {item_idx}/{sid}/b{bi}: {e}", file=sys.stderr)
                parsed = {}

            all_parsed.update(parsed)

        # Fallback + cache write
        for mi, m in enumerate(messages):
            tid = f"{si}_{mi}"
            ck = f"{sid}||{tid}"
            if ck not in cache and tid not in all_parsed:
                orig_text = m.get('content', '').strip()[:80]
                all_parsed[tid] = f"[auto] {orig_text}" if orig_text else "[auto] no topic"
            if ck not in cache and tid in all_parsed:
                cache[ck] = all_parsed[tid]

        # Print the topics of this session
        n_qwen = sum(1 for tid in all_tids
                     if tid in all_parsed and not all_parsed.get(tid, '').startswith('[auto]'))
        print(f"  [TurnTopics] item={item_idx} sid={sid}: Qwen={n_qwen}/{len(all_tids)}", file=sys.stderr)
        for tid in all_tids:
            if tid in all_parsed:
                marker = '' if all_parsed.get(tid, '').startswith('[auto]') else ' ✓'
                print(f"    {tid}{marker} → {all_parsed[tid][:100]}", file=sys.stderr)

        # Overall progress
        done = idx + 1
        pct = 100 * done / total_sessions
        elapsed = time.time() - t_start
        eta = elapsed / done * (total_sessions - done) if done > 0 else 0
        print(f"  [TurnTopics OVERALL] {done}/{total_sessions} sessions ({pct:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", file=sys.stderr, flush=True)

        if done % save_every == 0 or done >= total_sessions:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

        if done % 100 == 0:
            torch.cuda.empty_cache()

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"[INFO] Turn topics generated in {total_time/60:.1f} min, cached to {cache_path}")

    del model
    torch.cuda.empty_cache()
    return cache


# ═══════════════════════════════════════════
# Topic cluster construction
# ═══════════════════════════════════════════
def _build_topic_clusters_v3(item, turn_topics_cache, embed_fn):
    """
    Build topic clusters for every session of an item:
    1. Read the per-turn topic text from the cache
    2. Encode the topics with MiniLM
    3. Merge turns with similar topics using Agglomerative Clustering
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    topic_clusters = []
    cluster_idx = 0

    for si, sid in enumerate(item['haystack_session_ids']):
        msgs = item['haystack_sessions'][si]
        if not msgs:
            continue

        # Collect the topics of all turns in this session
        turn_topics = {}
        for mi in range(len(msgs)):
            tid = f"{si}_{mi}"
            ck = f"{sid}||{tid}"
            topic_text = turn_topics_cache.get(ck, '').strip()
            if not topic_text:
                topic_text = msgs[mi].get('content', '')[:80].strip()
            turn_topics[tid] = topic_text

        if not turn_topics:
            cluster_idx += 1
            continue

        tids_in_session = list(turn_topics.keys())
        topic_texts = [turn_topics[tid] for tid in tids_in_session]

        topic_embs = embed_fn(topic_texts)
        if topic_embs.ndim == 1:
            topic_embs = topic_embs.reshape(1, -1)

        n_turns = len(tids_in_session)
        if n_turns >= 3:
            sim_matrix = cosine_similarity(topic_embs)
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
            sim = cosine_similarity(topic_embs)[0, 1] if topic_embs.shape[0] == 2 else 0.0
            labels = [0, 0] if sim > 0.55 else [0, 1]
        else:
            labels = [0]

        cluster_groups = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_groups[int(label)].append((tids_in_session[i], topic_texts[i], topic_embs[i]))

        for label, members in cluster_groups.items():
            member_tids = [m[0] for m in members]
            member_texts = [m[1] for m in members]
            member_embs = np.array([m[2] for m in members])
            best_text = max(member_texts, key=len) if member_texts else ''
            topic_clusters.append({
                'id': f'topic_{cluster_idx}',
                'text': best_text,
                'turns': member_tids,
                'session': si,
                'sess_id': sid,
                'embedding': member_embs.mean(axis=0) if len(member_embs) > 0 else np.zeros(384, dtype=np.float32)
            })
            cluster_idx += 1

    return topic_clusters


# ═══════════════════════════════════════════
# Graph construction v3 update (Topic Cluster instead of Event)
# ═══════════════════════════════════════════
def build_graph_v3_update(item, summaries, turn_topics_cache, embed_fn):
    """
    L3 Session  ← session_summary
    L2 Turn     ← messages (role + content)
    L1 Topic Cluster ← Qwen-generated topic → MiniLM clustering

    Gold: turn ID of the message with has_answer=true
    """
    sessions = {}
    turns = {}
    sk_all = list(range(len(item['haystack_session_ids'])))
    turn_all = []
    turn_to_sess = {}
    gold = []

    for idx, sid in enumerate(item['haystack_session_ids']):
        msgs = item['haystack_sessions'][idx]
        date = item['haystack_dates'][idx] if idx < len(item['haystack_dates']) else ''
        summ = summaries.get(sid, {})

        sessions[idx] = {
            'id': idx, 'sid': sid, 'date': date,
            'text': summ.get('session_summary', ''),
            'turns': []
        }

        for mi, m in enumerate(msgs):
            tid = f"{idx}_{mi}"
            turns[tid] = {
                'id': tid, 'session': idx,
                'speaker': m.get('role', 'user'),
                'text': m.get('content', '')
            }
            sessions[idx]['turns'].append(tid)
            turn_all.append(tid)
            turn_to_sess[tid] = idx

            if m.get('has_answer', False):
                gold.append(tid)

    # Topic clusters (instead of events)
    topic_clusters = _build_topic_clusters_v3(item, turn_topics_cache, embed_fn)

    return {
        'sessions': sessions, 'turns': turns, 'topic_clusters': topic_clusters,
        'sk_all': sk_all, 'turn_all': turn_all, 'turn_to_sess': turn_to_sess,
        'gold': gold,
    }


# ═══════════════════════════════════════════
# Weight grid
# ═══════════════════════════════════════════
def generate_weight_grid_3d(step=0.10):
    n = int(1.0 / step)
    values = [round(i * step, 10) for i in range(n + 1)]
    combos = []
    for w1 in values:
        for w2 in values:
            w3 = 1.0 - w1 - w2
            if w3 < -0.0001 or w3 > 1.0001:
                continue
            if any(abs(w3 - v) < 0.0001 for v in values):
                combos.append((w1, w2, round(w3, 10)))
    return sorted(set(combos))


# ═══════════════════════════════════════════
# Cosine similarity + normalization
# ═══════════════════════════════════════════
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


# ═══════════════════════════════════════════
# Three-way hybrid recall + graph reranking (v3 update)
# ═══════════════════════════════════════════
def graph_hybrid_recall_rerank_v3(query, meta, weights, embed_fn,
                                   sess_k=3, topic_k=5, bm25_n=100, top_k=20):
    """
    Three-way recall → merge and deduplicate → graph-signal reranking

    Path 1: BM25(query, turn)              top-N         → keyword backbone
    Path 2: MiniLM(query, session_summary)  top-K session → all turns
    Path 3: MiniLM(query, topic_cluster)    top-K cluster → cluster turns
    """
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, bm25_turn) = meta

    w_bm25, w_turn, w_topic = weights
    q_emb = embed_fn(query)
    q_toks = set(tokenize(query))

    candidate_tids = set()

    # Path 1: BM25 top-N
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    for i in bm25_topn:
        candidate_tids.add(turn_all[i])

    # Path 2: MiniLM → top-K sessions → all turns
    if len(sess_embs) > 0:
        s_scores = _cosine_scores(q_emb, sess_embs)
        sess_topk = np.argsort(s_scores)[::-1][:sess_k]
        for si in sess_topk:
            sk = sk_all[si]
            for i, tid in enumerate(turn_all):
                if turn_to_sess[tid] == sk:
                    candidate_tids.add(tid)

    # Path 3: MiniLM → top-K topic clusters → cluster turns
    if len(topic_embs) > 0 and len(topic_to_turns) > 0:
        tc_scores = _cosine_scores(q_emb, topic_embs)
        tc_topk = np.argsort(tc_scores)[::-1][:topic_k]
        for tci in tc_topk:
            for tid in topic_to_turns.get(tci, []):
                candidate_tids.add(tid)

    if not candidate_tids:
        return []

    # Re-rank
    pool = list(candidate_tids)
    pool_indices = [turn_all.index(tid) for tid in pool]

    # BM25 score
    bm25_pool = {tid: float(bm25_scores_all[idx]) for tid, idx in zip(pool, pool_indices)}
    bm25_norm = _norm(bm25_pool)

    # Turn embedding
    cand_embs = turn_embs[pool_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    turn_pool = {pool[i]: float(t_cos[i]) for i in range(len(pool))}
    turn_norm = _norm(turn_pool)

    # Topic cluster score (broadcast to the turns in the pool)
    turn_topic_score = {}
    if len(topic_embs) > 0:
        tc_scores_all = _cosine_scores(q_emb, topic_embs)
        for tci in range(len(topic_to_turns)):
            score = float(tc_scores_all[tci])
            for tid in topic_to_turns.get(tci, []):
                turn_topic_score[tid] = max(turn_topic_score.get(tid, -999), score)
    topic_pool = {tid: turn_topic_score.get(tid, 0.0) for tid in pool}
    topic_norm = _norm(topic_pool)

    # Weighted sum
    final = {}
    for tid in pool:
        final[tid] = (w_bm25 * bm25_norm.get(tid, 0) +
                      w_turn * turn_norm.get(tid, 0) +
                      w_topic * topic_norm.get(tid, 0))

    ranked = sorted(final, key=final.get, reverse=True)[:top_k]
    return ranked


# ═══════════════════════════════════════════
# Ablation study: each of the three paths used independently (v3 meta: 8-tuple, no turn_texts_ordered)
# ═══════════════════════════════════════════
def ablation_bm25_only_v3(query, meta, bm25_n, top_k):
    """Ablation path 1: BM25 recall only → sort by BM25 score and take top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, bm25_turn) = meta
    q_toks = set(tokenize(query))
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    pool_tids = [turn_all[i] for i in bm25_topn]
    pool_scores = [float(bm25_scores_all[i]) for i in bm25_topn]
    ranked = sorted(zip(pool_tids, pool_scores), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


def ablation_sem_session_only_v3(query, meta, embed_fn, sess_k, top_k):
    """Ablation path 2: MiniLM→Session recall only → sort by turn-level cosine and take top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, bm25_turn) = meta
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


def ablation_sem_topic_only_v3(query, meta, embed_fn, topic_k, top_k):
    """Ablation path 3: MiniLM→Topic Cluster recall only → sort by turn-level cosine and take top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, bm25_turn) = meta
    if len(topic_embs) == 0 or len(topic_to_turns) == 0:
        return []
    q_emb = embed_fn(query)
    tc_scores = _cosine_scores(q_emb, topic_embs)
    tc_topk = np.argsort(tc_scores)[::-1][:topic_k]
    candidate_tids = []
    seen_tids = set()
    for tci in tc_topk:
        for tid in topic_to_turns.get(tci, []):
            if tid not in seen_tids:
                candidate_tids.append(tid)
                seen_tids.add(tid)
    if not candidate_tids:
        return []
    candidate_indices = [turn_all.index(tid) for tid in candidate_tids]
    cand_embs = turn_embs[candidate_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    ranked = sorted(zip(candidate_tids, t_cos), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


# ═══════════════════════════════════════════
# BM25 (normalized)
# ═══════════════════════════════════════════
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.lens = np.array([len(d) for d in docs])
        self.avgdl = np.mean(self.lens) if self.N > 0 else 1.0
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

    def search(self, query_tokens, top_k=20):
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
                    tf + self.k1 * (1 - self.b + self.b * self.lens[did] / max(self.avgdl, 1)))
        return scores


# ═══════════════════════════════════════════
# MiniLM Embedding
# ═══════════════════════════════════════════
def load_embedder():
    from transformers import AutoModel, AutoTokenizer
    import torch
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


def embed_retrieve(query_emb, doc_embs, top_k=20):
    scores = _cosine_scores(query_emb, doc_embs)
    idx = np.argsort(scores)[::-1][:top_k]
    return list(idx)


# ═══════════════════════════════════════════
# Turn-level baselines
# ═══════════════════════════════════════════
def rrf_retrieve_bm25_minilm_turn(query, turn_all, bm25_turn, q_emb, turn_embs,
                                  top_k=20, rrf_k=60):
    q_toks = tokenize(query)
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


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════
def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / max(len(gold), 1)


def mrr_score(ranked, gold):
    g = set(gold)
    for i, x in enumerate(ranked, 1):
        if x in g:
            return 1.0 / i
    return 0.0


def compute_f1(prediction, ground_truth):
    """Token-level F1"""
    pred_tokens = set(tokenize(str(prediction)))
    gt_tokens = set(tokenize(str(ground_truth)))
    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    if precision + recall == 0:
        return 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, recall


def compute_exact_match(prediction, ground_truth):
    """Exact match after normalization"""
    def norm(s):
        return str(s).lower().strip().rstrip('.')
    return 1.0 if norm(prediction) == norm(ground_truth) else 0.0


# ═══════════════════════════════════════════
# End-to-end answer generation
# ═══════════════════════════════════════════
def generate_answer_e2e(model, tokenizer, query, ranked_turns, turns_dict, max_retrieved=10):
    """
    Use the retrieved top-N turns as context and let Qwen generate the answer.
    Returns (answer_text, generation_time_seconds).
    """
    # Concatenate the top-N turns into the context
    top_turns = ranked_turns[:max_retrieved]
    turn_texts = []
    for tid in top_turns:
        turn = turns_dict.get(tid, {})
        speaker = turn.get('speaker', 'user')
        text = turn.get('text', '')
        turn_texts.append(f"[{speaker}]: {text}")

    context = '\n'.join(turn_texts)

    # Truncate the context
    if len(context) > 4000:
        context = context[:2000] + '\n...[truncated]...\n' + context[-2000:]

    prompt = (
        "Based on the following conversation history, answer the question. "
        "Answer as concisely as possible - just give the direct answer, no explanation.\n\n"
        "Conversation history:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    t0 = time.time()
    raw = _qwen_generate(model, tokenizer, prompt, max_new_tokens=100)
    gen_time = time.time() - t0

    # Clean up the output
    answer = raw.strip().split('\n')[0]  # take the first line
    return answer, gen_time


# ═══════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════
CAT_NAMES = {
    'single-session-user': 'single-user',
    'single-session-assistant': 'single-assist',
    'single-session-preference': 'single-pref',
    'multi-session': 'multi-session',
    'temporal-reasoning': 'temporal',
    'knowledge-update': 'knowledge',
}

TOP_K = 20  # retrieval top-k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='long-term-memory/longmemeval_s_cleaned.json')
    parser.add_argument('--qwen_path', type=str, default=None,
                        help='Path to Qwen2.5-7B-Instruct')
    parser.add_argument('--summary_cache', type=str, default='longmemeval_summaries_v3.json')
    parser.add_argument('--turn_topic_cache', type=str, default='turn_topics_cache_v3.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Small sample: use only the first N items')
    parser.add_argument('--weight_search', action='store_true',
                        help='Run a grid search for the best weights')
    parser.add_argument('--grid_step', type=float, default=0.10)
    parser.add_argument('--sem_session_k', type=int, default=3)
    parser.add_argument('--sem_topic_k', type=int, default=5)
    parser.add_argument('--sem_bm25_n', type=int, default=100)
    parser.add_argument('--end_to_end', action='store_true',
                        help='End-to-end answer generation + F1 evaluation')
    parser.add_argument('--no_timing', action='store_true',
                        help='Skip the timing statistics (for debugging)')
    args = parser.parse_args()

    # ═══ Load the data ═══
    print("[INFO] Loading dataset...")
    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    if args.max_items:
        data = data[:args.max_items]
        print(f"[INFO] Small-sample mode: {len(data)} items")

    # ═══ Phase 0a: Session Summaries ═══
    if args.qwen_path:
        summaries = load_or_generate_summaries(data, args.qwen_path, args.summary_cache)
    else:
        if not os.path.exists(args.summary_cache):
            print(f"[ERROR] Summary cache not found: {args.summary_cache}")
            print("  First run: add --qwen_path /path/to/Qwen2.5-7B-Instruct")
            return
        with open(args.summary_cache, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

    # ═══ Phase 0b: Turn Topics ═══
    if args.qwen_path:
        turn_topics_cache = load_or_generate_turn_topics_v3(
            data, args.qwen_path, args.turn_topic_cache, summaries
        )
    else:
        if not os.path.exists(args.turn_topic_cache):
            print(f"[ERROR] Turn topic cache not found: {args.turn_topic_cache}")
            print("  First run: add --qwen_path /path/to/Qwen2.5-7B-Instruct")
            return
        with open(args.turn_topic_cache, 'r', encoding='utf-8') as f:
            turn_topics_cache = json.load(f)

    embed = load_embedder()

    # ═══ Phase 1: build all graphs + precompute embeddings ═══
    t_phase1_start = time.time()
    all_items = []
    all_meta = []

    for i, item in enumerate(data):
        t_item_start = time.time() if not args.no_timing else 0
        print(f"[{i + 1}/{len(data)}] {item['question_id']} ({item.get('question_type','?')}) "
              f"sessions={len(item['haystack_sessions'])}")

        graph = build_graph_v3_update(item, summaries, turn_topics_cache, embed)
        sk_all = graph['sk_all']
        sessions = graph['sessions']
        turns = graph['turns']
        topic_clusters = graph['topic_clusters']
        turn_all = graph['turn_all']
        turn_to_sess = graph['turn_to_sess']
        gold = graph['gold']
        qtype = item.get('question_type', 'unknown')
        query = item.get('question', '')

        if not gold:
            print(f"  -> SKIP (no gold turns)", file=sys.stderr)
            continue

        # Session Embeddings
        sess_texts = [sessions[sk]['text'] or ' ' for sk in sk_all]
        sess_embs = embed(sess_texts)

        # Turn Embeddings
        turn_texts_ordered = []
        t2s = []
        for tid in turn_all:
            turn = turns[tid]
            txt = turn['text'].strip()
            turn_texts_ordered.append(
                f"{turn['speaker']}: {txt}" if turn['speaker'] and txt else (txt or ' ')
            )
            t2s.append(turn_to_sess[tid])
        print(f"  -> embedding {len(turn_texts_ordered)} turns...", flush=True, file=sys.stderr)
        turn_embs = embed(turn_texts_ordered) if turn_texts_ordered else np.zeros((0, 384), dtype=np.float32)

        # Turn BM25
        turn_tokens = [tokenize(t) for t in turn_texts_ordered]
        bm25_turn = BM25(turn_tokens)

        # Topic Cluster Embeddings
        topic_texts = [tc['text'] for tc in topic_clusters]
        print(f"  -> embedding {len(topic_texts)} topic clusters...", flush=True, file=sys.stderr)
        if topic_texts:
            topic_embs_raw = embed(topic_texts)
            if topic_embs_raw.ndim == 1:
                topic_embs_raw = topic_embs_raw.reshape(1, -1)
        else:
            topic_embs_raw = np.zeros((0, 384), dtype=np.float32)

        # topic_to_turns map
        topic_to_turns = {}
        for tci, tc in enumerate(topic_clusters):
            topic_to_turns[tci] = tc['turns']

        print(f"  -> sessions={len(sessions)}, turns={len(turns)}, topic_clusters={len(topic_clusters)}", file=sys.stderr)

        meta_idx = len(all_meta)
        all_meta.append((sk_all, sess_embs, turn_embs, topic_embs_raw, topic_to_turns,
                         turn_all, turn_to_sess, bm25_turn))
        all_items.append((meta_idx, query, gold, qtype, i))  # + item index for answer lookup

    t_phase1_end = time.time()
    total_qa = len(all_items)
    build_total_time = t_phase1_end - t_phase1_start
    build_time_per_q = build_total_time / max(total_qa, 1)
    print(f"\n[INFO] Phase 1 done: {total_qa} valid items in {build_total_time:.1f}s "
          f"({build_time_per_q:.2f}s per question)\n")

    if total_qa == 0:
        print("[ERROR] No valid items! Exiting.")
        return

    # ═══ Phase 2: grid search or fixed weights ═══
    grid_results = None
    weights = (0.50, 0.25, 0.25)  # default: w_bm25, w_turn, w_topic
    retrieve_fn = lambda q, m, w: graph_hybrid_recall_rerank_v3(
        q, m, w, embed, args.sem_session_k, args.sem_topic_k, args.sem_bm25_n, TOP_K)
    mode_name = 'Graph_TurnTopicRecall'

    if args.weight_search:
        weight_grid = generate_weight_grid_3d(args.grid_step)
        print(f"{'='*80}")
        print(f"WEIGHT GRID SEARCH: {len(weight_grid)} combinations (step={args.grid_step})")
        print(f"{'='*80}")

        all_grid_results = []
        best = {'weights': None, 'r10': 0.0}
        report_every = max(1, len(weight_grid) // 20)

        for gi, w in enumerate(weight_grid):
            r1_vals, r3_vals, r5_vals, r10_vals, mrr_vals = [], [], [], [], []
            for item_idx, query, gold, qtype, _ in all_items:
                meta = all_meta[item_idx]
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
                best = {'weights': w, 'r1': avg_r1,
                        'r3': avg_r3, 'r5': avg_r5,
                        'r10': avg_r10, 'mrr': avg_mrr}

            if gi % report_every == 0 or gi == len(weight_grid) - 1:
                pct = 100 * (gi + 1) / len(weight_grid)
                marker = ' ★' if is_new_best else '  '
                print(f"  [{gi+1:>4}/{len(weight_grid)} {pct:5.1f}%]{marker} "
                      f"R@10={avg_r10:.4f} (best={best['r10']:.4f})")

        weights = best['weights']
        all_grid_results.sort(key=lambda x: x['R@10'], reverse=True)
        grid_results = all_grid_results  # keep all results

        print(f"\n[Grid Search] BEST -> R@10={best['r10']:.4f}  "
              f"R@5={best['r5']:.4f}  MRR={best['mrr']:.4f}")
        print(f"  Weights: {dict(zip(['bm25','turn','topic'], [f'{v:.2f}' for v in weights]))}")
        print(f"\n[Grid Search] All {len(all_grid_results)} combinations (sorted by R@10):")
        print(f"  {'Rank':<5} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}   bm25    turn    topic")
        for rank, r in enumerate(all_grid_results, 1):
            ws = r['weights']
            marker = ' ★' if (ws['bm25'] == round(weights[0], 2) and
                              ws['turn'] == round(weights[1], 2) and
                              ws['topic'] == round(weights[2], 2)) else '  '
            print(f"  {rank:<5} {r['R@1']:>8.4f} {r['R@3']:>8.4f} {r['R@5']:>8.4f} "
                  f"{r['R@10']:>8.4f} {r['MRR']:>8.4f}   "
                  f"{ws['bm25']:.2f}   {ws['turn']:.2f}   {ws['topic']:.2f}{marker}")
    else:
        print(f"[INFO] Default weights: bm25={weights[0]:.2f} turn={weights[1]:.2f} topic={weights[2]:.2f}")

    # ═══ Phase 3: retrieval evaluation (turn level) + timing ═══
    ablation_methods = ['Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    methods = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25'] + ablation_methods
    overall = {m: defaultdict(list) for m in methods}
    per_cat = {m: defaultdict(lambda: defaultdict(list)) for m in methods}
    retrieval_times = defaultdict(list)  # per-method per-query time

    for eval_idx, (item_idx, query, gold, qtype, data_idx) in enumerate(all_items):
        meta = all_meta[item_idx]
        (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
         turn_all, turn_to_sess, bm25_turn) = meta

        # Graph
        t0 = time.time()
        g_ranked = retrieve_fn(query, meta, weights)
        retrieval_times[mode_name].append(time.time() - t0)

        q_emb = embed(query)
        q_toks = set(tokenize(query))

        # BM25+MiniLM RRF
        t0 = time.time()
        rrf_ranked = rrf_retrieve_bm25_minilm_turn(query, turn_all, bm25_turn, q_emb, turn_embs,
                                                   top_k=TOP_K, rrf_k=60)
        retrieval_times['BM25+MiniLM_RRF'].append(time.time() - t0)

        # MiniLM
        t0 = time.time()
        m_idx = embed_retrieve(q_emb, turn_embs, top_k=TOP_K)
        m_ranked = [turn_all[i] for i in m_idx]
        retrieval_times['MiniLM'].append(time.time() - t0)

        # BM25
        t0 = time.time()
        b_idx = bm25_turn.search(q_toks, top_k=TOP_K)
        b_ranked = [turn_all[i] for i in b_idx]
        retrieval_times['BM25'].append(time.time() - t0)

        # ── Ablation: each of the three paths used independently ──
        t0 = time.time()
        abl_bm25_ranked = ablation_bm25_only_v3(query, meta, args.sem_bm25_n, TOP_K)
        retrieval_times['Abl_BM25'].append(time.time() - t0)

        t0 = time.time()
        abl_sess_ranked = ablation_sem_session_only_v3(query, meta, embed, args.sem_session_k, TOP_K)
        retrieval_times['Abl_SemSession'].append(time.time() - t0)

        t0 = time.time()
        abl_topic_ranked = ablation_sem_topic_only_v3(query, meta, embed, args.sem_topic_k, TOP_K)
        retrieval_times['Abl_SemTopic'].append(time.time() - t0)

        for method, ranked in [(mode_name, g_ranked), ('BM25+MiniLM_RRF', rrf_ranked),
                               ('MiniLM', m_ranked), ('BM25', b_ranked),
                               ('Abl_BM25', abl_bm25_ranked),
                               ('Abl_SemSession', abl_sess_ranked),
                               ('Abl_SemTopic', abl_topic_ranked)]:
            for k in [1, 3, 5, 10]:
                overall[method][f'R@{k}'].append(recall_at_k(ranked, gold, k))
                per_cat[method][qtype][f'R@{k}'].append(recall_at_k(ranked, gold, k))
            overall[method]['MRR'].append(mrr_score(ranked, gold))
            per_cat[method][qtype]['MRR'].append(mrr_score(ranked, gold))

        # Progress
        if (eval_idx + 1) % 50 == 0 or eval_idx + 1 == total_qa:
            pct = 100 * (eval_idx + 1) / total_qa
            print(f"  [Retrieval Eval] {eval_idx + 1}/{total_qa} ({pct:.1f}%)", flush=True)

    # ═══ Phase 4: end-to-end generation + F1 (main method + baselines + ablations) ═══
    baseline_e2e_methods = ['BM25+MiniLM_RRF', 'MiniLM', 'BM25']
    e2e_methods = [mode_name] + baseline_e2e_methods + ablation_methods
    e2e_all = {}
    if args.end_to_end and args.qwen_path:
        import torch
        print(f"\n{'='*80}")
        print("END-TO-END ANSWER GENERATION + F1 EVALUATION")
        print(f"{'='*80}")
        e2e_model, e2e_tokenizer = _load_qwen_model(args.qwen_path)

        # Keep a separate list for every method
        e2e_f1 = {m: [] for m in e2e_methods}
        e2e_em = {m: [] for m in e2e_methods}
        e2e_gen_time = {m: [] for m in e2e_methods}
        e2e_total_time = {m: [] for m in e2e_methods}  # retrieval + generation

        total_with_answers = sum(1 for _, _, _, _, di in all_items if data[di].get('answer'))
        ab_evaluated = 0

        for e2e_idx, (item_idx, query, gold, qtype, data_idx) in enumerate(all_items):
            item = data[data_idx]
            gt_answer = item.get('answer', '')
            if not gt_answer:
                continue

            meta = all_meta[item_idx]
            (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
             turn_all, turn_to_sess, bm25_turn) = meta
            # Build the turns dict (used to assemble the context, reusable by all methods)
            graph = build_graph_v3_update(item, summaries, turn_topics_cache, embed)
            turns_dict = graph['turns']

            q_emb = embed(query)
            q_toks = set(tokenize(query))

            # Collect the ranked results of every method
            ranked_map = {}
            ranked_map[mode_name] = retrieve_fn(query, meta, weights)
            ranked_map['BM25+MiniLM_RRF'] = rrf_retrieve_bm25_minilm_turn(
                query, turn_all, bm25_turn, q_emb, turn_embs, top_k=TOP_K, rrf_k=60)
            m_idx = embed_retrieve(q_emb, turn_embs, top_k=TOP_K)
            ranked_map['MiniLM'] = [turn_all[i] for i in m_idx]
            b_idx = bm25_turn.search(q_toks, top_k=TOP_K)
            ranked_map['BM25'] = [turn_all[i] for i in b_idx]
            ranked_map['Abl_BM25'] = ablation_bm25_only_v3(query, meta, args.sem_bm25_n, TOP_K)
            ranked_map['Abl_SemSession'] = ablation_sem_session_only_v3(query, meta, embed, args.sem_session_k, TOP_K)
            ranked_map['Abl_SemTopic'] = ablation_sem_topic_only_v3(query, meta, embed, args.sem_topic_k, TOP_K)

            for em_name in e2e_methods:
                ranked = ranked_map[em_name]
                pred_answer, gen_t = generate_answer_e2e(
                    e2e_model, e2e_tokenizer, query, ranked, turns_dict, max_retrieved=10
                )
                ret_t = retrieval_times[em_name][e2e_idx] if e2e_idx < len(retrieval_times[em_name]) else 0.0
                f1, recall = compute_f1(pred_answer, gt_answer)
                em = compute_exact_match(pred_answer, gt_answer)
                e2e_f1[em_name].append(f1)
                e2e_em[em_name].append(em)
                e2e_gen_time[em_name].append(gen_t)
                e2e_total_time[em_name].append(ret_t + gen_t)

            ab_evaluated += 1
            if ab_evaluated % 50 == 0 or ab_evaluated == total_with_answers:
                parts = []
                for em_name in e2e_methods:
                    f = np.mean(e2e_f1[em_name]) if e2e_f1[em_name] else 0.0
                    parts.append(f"{em_name}:F1={f:.4f}")
                print(f"  [E2E] {ab_evaluated} QA: " + " | ".join(parts), file=sys.stderr)

        for em_name in e2e_methods:
            avg_f1 = float(np.mean(e2e_f1[em_name])) if e2e_f1[em_name] else 0.0
            avg_em = float(np.mean(e2e_em[em_name])) if e2e_em[em_name] else 0.0
            avg_gt = float(np.mean(e2e_gen_time[em_name])) if e2e_gen_time[em_name] else 0.0
            avg_total = float(np.mean(e2e_total_time[em_name])) if e2e_total_time[em_name] else 0.0
            e2e_all[em_name] = {
                'f1': avg_f1, 'exact_match': avg_em,
                'avg_generation_time_s': avg_gt,
                'avg_total_pipeline_time_s': avg_total,
                'num_evaluated': len(e2e_f1[em_name])
            }

        print(f"\n[E2E Results]")
        for em_name in e2e_methods:
            avg_f1 = float(np.mean(e2e_f1[em_name])) if e2e_f1[em_name] else 0.0
            avg_em = float(np.mean(e2e_em[em_name])) if e2e_em[em_name] else 0.0
            avg_gt = float(np.mean(e2e_gen_time[em_name])) if e2e_gen_time[em_name] else 0.0
            avg_total = float(np.mean(e2e_total_time[em_name])) if e2e_total_time[em_name] else 0.0
            print(f"  {em_name}: F1={avg_f1:.4f}  EM={avg_em:.4f}  "
                  f"retrieval+gen={avg_total:.2f}s  gen_only={avg_gt:.2f}s")

        del e2e_model
        torch.cuda.empty_cache()

    elif args.end_to_end and not args.qwen_path:
        print("[WARN] --end_to_end requires --qwen_path. Skipping E2E.", file=sys.stderr)

    # ═══ Phase 5: print + save ═══
    print("\n" + "=" * 120)
    suffix = " (grid-best)" if args.weight_search else ""
    all_display_methods = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
                           'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    print(f"RESULTS ({total_qa} items, {len(data)} total)")
    print("=" * 120)
    metrics = ['R@1', 'R@3', 'R@5', 'R@10', 'MRR']
    header = f"{'Metric':<8}" + "".join(f"{m:>16}" for m in all_display_methods)
    print(header)
    print("-" * 120)
    for met in metrics:
        vals = [np.mean(overall[m][met]) for m in methods]
        print(f"{met:<8}" + "".join(f"{v:>16.4f}" for v in vals))

    # E2E summary (main method + baselines + ablations)
    if e2e_all:
        print("\n" + "-" * 120)
        print("END-TO-END (F1 / EM / Time)")
        e2e_display = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
                       'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
        header2 = f"{'Metric':<14}" + "".join(f"{m:>16}" for m in e2e_display)
        print(header2)
        print("-" * 120)
        for met_name, met_key in [('F1', 'f1'), ('EM', 'exact_match')]:
            row = f"{met_name:<14}"
            for em_name in e2e_display:
                val = e2e_all.get(em_name, {}).get(met_key, 0.0)
                row += f"{val:>16.4f}"
            print(row)
        # Total pipeline time (retrieval + generation)
        row = f"{'ret+gen(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_total_pipeline_time_s', 0.0)
            row += f"{val:>16.2f}"
        print(row)
        # Generation time only
        row = f"{'gen_only(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_generation_time_s', 0.0)
            row += f"{val:>16.2f}"
        print(row)

    # Timing statistics
    print("\n" + "-" * 120)
    print("TIME STATISTICS")
    print(f"{'Metric':<30} {'Value':>20}")
    print("-" * 60)
    print(f"{'Pre-build total time':<30} {build_total_time:>20.1f}s")
    print(f"{'Graph build time per question':<30} {build_time_per_q:>20.2f}s")
    for m in methods:
        t = retrieval_times[m]
        avg_t = np.mean(t) if t else 0
        print(f"{'Retrieval time per q ('+m+')':<30} {avg_t:>20.4f}s")
    if e2e_all:
        main_data = e2e_all.get(mode_name, {})
        gen_t = main_data.get('avg_generation_time_s', 0)
        total_per_q = build_time_per_q + np.mean(retrieval_times[mode_name]) + gen_t
        print(f"{'E2E generation time per q':<30} {gen_t:>20.2f}s")
        print(f"{'Total time per question (approx)':<30} {total_per_q:>20.2f}s")

    # Per category
    print("\n" + "-" * 120)
    print("PER-CATEGORY (R@5)")
    print(f"{'Category':<18}" + "".join(f"{m:>16}" for m in all_display_methods))
    print("-" * 120)
    qtype_order = sorted(per_cat[mode_name].keys(),
                         key=lambda qt: np.mean(per_cat[mode_name][qt].get('R@5', [0])),
                         reverse=True)
    for qt in qtype_order:
        label = CAT_NAMES.get(qt, qt[:16])
        row = f"{label:<18}"
        for m in methods:
            vals = per_cat[m][qt].get('R@5', [])
            row += f"{np.mean(vals):>16.4f}" if vals else f"{'N/A':>16}"
        print(row)

    # ═══ Save ═══
    out = {
        'version': 'graph_hybrid_recall_v3_update',
        'dataset': 'longmemeval_s_cleaned',
        'total_qa': total_qa,
        'num_total': len(data),
        'top_k': TOP_K,
        'weight_search': args.weight_search,
        'weights': {k: float(v) for k, v in zip(['bm25', 'turn', 'topic'], weights)},
        'sem_session_k': args.sem_session_k,
        'sem_topic_k': args.sem_topic_k,
        'sem_bm25_n': args.sem_bm25_n,
        'has_stemmer': _HAS_STEMMER,
        'pre_build_time_s': round(build_total_time, 1),
        'overall': {
            m: {k: float(np.mean(v)) if v else 0.0 for k, v in ov.items()}
            for m, ov in overall.items()
        },
        'ablation': {
            'Abl_BM25': 'Path 1 only: BM25 query→turn recall',
            'Abl_SemSession': 'Path 2 only: MiniLM(query, session_summary) recall',
            'Abl_SemTopic': 'Path 3 only: MiniLM(query, topic_cluster) recall',
        },
        'per_category_r5': {},
        'per_category_r10': {},
        'timing': {
            'build_total_time_s': round(build_total_time, 1),
            'build_time_per_q_s': build_time_per_q,
            'retrieval_time_per_q_s': {
                m: float(np.mean(retrieval_times[m])) if retrieval_times[m] else 0.0
                for m in methods
            }
        }
    }

    if e2e_all:
        out['e2e_all'] = e2e_all

    for qt in qtype_order:
        out['per_category_r5'][qt] = {}
        out['per_category_r10'][qt] = {}
        for m in methods:
            vals5 = per_cat[m][qt].get('R@5', [])
            vals10 = per_cat[m][qt].get('R@10', [])
            out['per_category_r5'][qt][m] = float(np.mean(vals5)) if vals5 else None
            out['per_category_r10'][qt][m] = float(np.mean(vals10)) if vals10 else None

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
