"""
LoCoMo experiment: three-path hybrid recall + graph-signal re-ranking
============================================================================
Overview
--------
HG-Mem on the LoCoMo benchmark. The pipeline builds a hierarchical memory
graph (Session -> Topic-Cluster -> Turn) per conversation, recalls candidate
turns through three complementary paths (BM25, MiniLM->session,
MiniLM->topic cluster), merges and de-duplicates them, and then re-ranks the
candidates using three normalised signals (BM25 lexical, turn-level semantic,
topic-aware semantic).

Update notes
------------
  1. MRR@20 -> MRR@10: every retrieval top_k becomes 10.
  2. Event-level -> Turn-Topic level: Qwen-7B produces a topic phrase for
     every turn; turns within the same session that share similar topics are
     merged into a topic cluster.
  3. Third recall path: MiniLM(query, topic_cluster) -> only turns inside the
     matched cluster, reducing noise.

Three recall paths
------------------
  Path 1: BM25(query, turn)              top-100  (lexical backbone)
  Path 2: MiniLM(query, session_summary)  top-3    -> all turns of the session
                                                 (semantic -> coarse topic)
  Path 3: MiniLM(query, topic_cluster)    top-5    -> only turns inside the
                                                 cluster (semantic -> fine topic)
Merge + de-duplicate, then re-rank with
    w_bm25 * bm25_norm + w_turn * turn_norm + w_topic * topic_norm
and return the top-10 turns.

Gold evidence
-------------
  Evidence is taken directly from the `evidence` field of each QA pair
  (the list of turn IDs).

Baselines (turn level)
----------------------
  - BM25
  - MiniLM (all-MiniLM-L6-v2 cosine similarity)
  - BM25 + MiniLM with Reciprocal Rank Fusion (RRF)

Usage
-----
  First run (generates turn topics via Qwen):
    python graph_vs_baselines_LoCoMo.py \
        --qwen_path /path/to/Qwen2.5-7B-Instruct --weight_search
  Subsequent runs (loads the cache, Qwen not required):
    python graph_vs_baselines_LoCoMo.py --weight_search
  End-to-end evaluation (R@10 -> Qwen generates answer -> F1/EM):
    python graph_vs_baselines_LoCoMo.py \
        --qwen_path /path/to/Qwen2.5-7B-Instruct \
        --weight_search --end_to_end
"""

import argparse
import json, sys, io, math, os, time
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


# ============================================================================
# Tokenisation helper
# ============================================================================
_STOPS = {'the','a','an','is','are','was','were','in','on','at','to','for',
          'of','with','and','or','by','from','it','its','i','you','he','she',
          'we','they','my','your','his','her','our','their','me','him','us',
          'them','that','this','these','those','be','been','being','have','has',
          'had','do','does','did','will','would','can','could','should','may',
          'might','not','no','but','if','so','very','just','about','also',
          'what','when','where','who','how','why','which','whom'}

def tokenize(text):
    """Lower-case, strip punctuation, drop stop-words and 1-char tokens."""
    t = text.lower()
    for ch in '?!.,;:\"\'()[]{}-\n\r':
        t = t.replace(ch, ' ')
    return [w for w in t.split() if w not in _STOPS and len(w) > 1]


# ============================================================================
# Turn-id sort key (D{session:1}:{turn:1})
# ============================================================================
def _dia_sort_key(dia_id):
    """Convert a turn-id such as 'D1:3' into the numeric pair (1, 3)."""
    try:
        parts = dia_id.replace('D', '').split(':')
        return (int(parts[0]), int(parts[1]))
    except:
        return (0, 0)


# ============================================================================
# Qwen turn-topic generation (per-session batch)
# ============================================================================
def _qwen_generate(model, tokenizer, prompt, max_new_tokens=256):
    """Single Qwen generation call."""
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


def _parse_turn_topics(raw_output, expected_dia_ids):
    """
    Parse the Qwen output where every line is of the form
    `turn_id || topic phrase` or `turn_id: topic phrase`.
    Returns a dictionary {dia_id: topic_phrase}.
    """
    topics = {}
    for line in raw_output.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Try the "||" separator first.
        if '||' in line:
            parts = line.split('||', 1)
            tid = parts[0].strip()
            phrase = parts[1].strip() if len(parts) > 1 else ''
        elif ': ' in line:
            # Try ": " next, but be careful with the inner ':' of "D1:1".
            # We only accept a turn-id matching the pattern D<num>:<num>.
            import re
            m = re.match(r'(D\d+:\d+)\s*[:：]\s*(.*)', line)
            if m:
                tid = m.group(1)
                phrase = m.group(2).strip()
            else:
                continue
        else:
            continue

        if tid not in expected_dia_ids:
            continue
        phrase = phrase.strip().lstrip('-•·').strip()
        if len(phrase) >= 3:
            topics[tid] = phrase

    return topics


def _build_session_prompt(session_messages, sk):
    """Build the per-session topic-generation prompt."""
    lines = []
    dia_ids = []
    for m in session_messages:
        if 'dia_id' not in m:
            continue
        did = m['dia_id']
        speaker = m.get('speaker', 'unknown')
        content = m.get('text', '').strip()
        if len(content) > 500:
            content = content[:250] + "...[truncated]..." + content[-250:]
        lines.append(f"{did} [{speaker}]: {content}")
        dia_ids.append(did)

    if not lines:
        return '', []

    # Sort by dia_id so the model sees the conversation in order.
    pairs = list(zip(dia_ids, lines))
    pairs.sort(key=lambda x: _dia_sort_key(x[0]))
    sorted_dia_ids = [p[0] for p in pairs]
    sorted_lines = [p[1] for p in pairs]

    formatted = '\n'.join(sorted_lines)
    expected_ids_str = ', '.join(sorted_dia_ids)

    prompt = (
        f"For each turn in this conversation session {sk}, write ONE brief topic phrase "
        f"(5-10 words) describing the main subject of that turn. Be specific - include "
        f"names, entities, and key information mentioned.\n\n"
        f"Output format (one per line, EXACTLY):\n"
        f"turn_id || topic phrase\n\n"
        f"Expected turn IDs: {expected_ids_str}\n\n"
        f"Conversation:\n{formatted}\n\n"
        f"Topics:"
    )
    return prompt, sorted_dia_ids


def load_or_generate_turn_topics(data, qwen_path, cache_path):
    """
    Load the cached turn topics or incrementally generate missing entries.
    Cache key:   f"{sample_id}||{dia_id}"
    Cache value: topic_phrase (string)
    """
    cache = {}
    if os.path.exists(cache_path):
        print(f"[INFO] Loading existing turn-topic cache from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] {len(cache)} turns already cached")

    # Collect every turn whose topic is still missing.
    # Group them by (sample_id, session_key) so Qwen can be prompted per session.
    all_needed = {}  # composite_key -> (sample_id, session_key, messages)
    for item in data:
        sid = item['sample_id']
        conv = item['conversation']
        sk_all = sorted(
            [k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')],
            key=lambda x: int(x.split('_')[1])
        )
        for sk in sk_all:
            for m in conv[sk]:
                if 'dia_id' not in m:
                    continue
                cache_key = f"{sid}||{m['dia_id']}"
                if cache_key not in cache:
                    composite_key = f"{sid}||{sk}"
                    if composite_key not in all_needed:
                        all_needed[composite_key] = (sid, sk, [])
                    all_needed[composite_key][2].append(m)

    if not all_needed:
        print("[INFO] All turn topics already cached!")
        return cache

    total_sessions = len(all_needed)
    total_turns = sum(len(v[2]) for v in all_needed.values())
    print(f"[INFO] {total_sessions} sessions ({total_turns} turns) need topic generation (Qwen: {qwen_path})")

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

    needed_list = list(all_needed.values())
    t_start = time.time()
    save_every = 5  # Save the cache every 5 sessions.

    BATCH_SIZE = 20  # At most 20 turns per Qwen prompt to avoid the model "skipping" turns.

    for idx, (sample_id, sk, messages) in enumerate(needed_list):
        # Deduplicate by dia_id, then sort by dia_id.
        seen_did = set()
        unique_msgs = []
        for m in messages:
            if m['dia_id'] not in seen_did:
                seen_did.add(m['dia_id'])
                unique_msgs.append(m)
        unique_msgs.sort(key=lambda m: _dia_sort_key(m['dia_id']))

        if not unique_msgs:
            continue

        # Split into batches of <= BATCH_SIZE turns.
        all_parsed = {}
        num_batches = (len(unique_msgs) + BATCH_SIZE - 1) // BATCH_SIZE

        for bi in range(num_batches):
            batch_msgs = unique_msgs[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            prompt, expected_ids = _build_session_prompt(batch_msgs, sk)

            if not expected_ids:
                continue

            # Truncate overly long prompts to avoid hitting Qwen's context limit.
            if len(prompt) > 8000:
                prompt = prompt[:4000] + "\n\n...[truncated]...\n\n" + prompt[-4000:]

            batch_label = f"{sk}[b{bi+1}/{num_batches}]" if num_batches > 1 else sk
            try:
                mnt = max(300, len(expected_ids) * 40)
                raw = _qwen_generate(model, tokenizer, prompt, max_new_tokens=mnt)
                parsed = _parse_turn_topics(raw, set(expected_ids))
                if num_batches > 1:
                    print(f"    [batch {bi+1}/{num_batches}] Qwen={len(parsed)}/{len(expected_ids)}", file=sys.stderr)
            except Exception as e:
                print(f"  [WARN] Qwen failed for {sample_id}/{batch_label}: {e}", file=sys.stderr)
                parsed = {}
                raw = ''

            all_parsed.update(parsed)

        # Fallback for turns that were not produced by Qwen:
        # use the first 80 chars of the original turn as a stand-in topic.
        for m in unique_msgs:
            did = m['dia_id']
            cache_key = f"{sample_id}||{did}"
            if cache_key not in cache and did not in all_parsed:
                orig_text = m.get('text', '').strip()[:80]
                all_parsed[did] = f"[auto] {orig_text}" if orig_text else "[auto] no topic"
            if cache_key not in cache and did in all_parsed:
                cache[cache_key] = all_parsed[did]

        # Print the topics produced for the current session.
        n_qwen = sum(1 for m in unique_msgs
                     if m['dia_id'] in all_parsed and not all_parsed[m['dia_id']].startswith('[auto]'))
        print(f"  [TurnTopics] {sample_id}/{sk}: Qwen={n_qwen}/{len(unique_msgs)} topics", file=sys.stderr)
        for m in unique_msgs:
            did = m['dia_id']
            if did in all_parsed:
                marker = '' if all_parsed[did].startswith('[auto]') else ' OK'
                print(f"    {did}{marker} -> {all_parsed[did][:100]}", file=sys.stderr)

        # Overall progress.
        done = idx + 1
        pct = 100 * done / total_sessions
        elapsed = time.time() - t_start
        eta = elapsed / done * (total_sessions - done) if done > 0 else 0
        print(f"  [TurnTopics OVERALL] {done}/{total_sessions} sessions ({pct:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        # Incremental save.
        if done % save_every == 0 or done >= total_sessions:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

        # Periodically release GPU memory.
        if done % 50 == 0:
            torch.cuda.empty_cache()

    # Final save.
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"[INFO] Turn topics generated in {total_time/60:.1f} min, cached to {cache_path}")

    del model
    torch.cuda.empty_cache()
    return cache


# ============================================================================
# Graph construction (v2 update; turn-topic clusters)
# ============================================================================
def build_graph_v2_update(item, turn_topics_cache, embed_fn):
    """
    Build the hierarchical memory graph for one conversation.

    Graph structure:
      L3 Session    <- session_summary
      L2 Turn       <- original dialogue text (speaker + text)
      L1 Topic Cl. <- Qwen-generated per-turn topics, clustered by MiniLM

    Each topic cluster is represented as:
      {id, text, turns[tid, ...], session, embedding}

    Returns
    -------
    graph = {sessions, turns, topic_clusters, sk_all, turn_all, turn_to_sess}
    """
    conv = item['conversation']
    sample_id = item['sample_id']
    sk_all = sorted(
        [k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')],
        key=lambda x: int(x.split('_')[1])
    )

    # ---- L3 Session ----
    ss = item.get('session_summary', {})
    sessions = {}
    for sk in sk_all:
        sn = int(sk.split('_')[1])
        sessions[sk] = {
            'id': sk,
            'text': ss.get(f'session_{sn}_summary', ''),
            'turns': []
        }

    # ---- L2 Turn ----
    turns = {}
    for sk in sk_all:
        msgs = conv[sk]
        for m in msgs:
            if 'dia_id' not in m:
                continue
            turns[m['dia_id']] = {
                'id': m['dia_id'], 'session': sk,
                'speaker': m.get('speaker', ''), 'text': m.get('text', '')
            }
            sessions[sk]['turns'].append(m['dia_id'])

    # ---- Turn-level metadata ----
    turn_all = []
    turn_to_sess = {}
    for sk in sk_all:
        for m in conv[sk]:
            if 'dia_id' in m:
                tid = m['dia_id']
                turn_all.append(tid)
                turn_to_sess[tid] = sk

    # ---- L1 Topic Clusters ----
    topic_clusters = _build_topic_clusters(
        item, sk_all, turns, turn_topics_cache, embed_fn, sample_id
    )

    return {
        'sessions': sessions, 'turns': turns, 'topic_clusters': topic_clusters,
        'sk_all': sk_all,
        'turn_all': turn_all, 'turn_to_sess': turn_to_sess,
    }


def _build_topic_clusters(item, sk_all, turns, turn_topics_cache, embed_fn, sample_id):
    """
    Build the topic clusters for every session of one conversation:

    1. Load the per-turn topic text from the cache (fallback: first 80 chars
       of the original turn).
    2. Encode the topic texts with MiniLM.
    3. Within each session, use Agglomerative Clustering to merge turns that
       share topics.

    Each cluster is returned as
      {id, text, turns[tid, ...], session, embedding}
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    topic_clusters = []
    cluster_idx = 0
    total_sessions = len(sk_all)

    for si, sk in enumerate(sk_all):
        # Collect the per-turn topics for this session.
        turn_topics = {}  # dia_id -> topic_text
        for tid, tdata in turns.items():
            if tdata['session'] != sk:
                continue
            cache_key = f"{sample_id}||{tid}"
            topic_text = turn_topics_cache.get(cache_key, '').strip()
            if not topic_text:
                # Fallback: use the first 80 chars of the original turn.
                topic_text = turns[tid].get('text', '')[:80].strip()
            turn_topics[tid] = topic_text

        if not turn_topics:
            cluster_idx += 1
            continue

        tids_in_session = list(turn_topics.keys())
        topic_texts = [turn_topics[tid] for tid in tids_in_session]

        # Encode the topic texts with MiniLM.
        if len(topic_texts) == 0:
            continue
        topic_embs = embed_fn(topic_texts)
        if topic_embs.ndim == 1:
            topic_embs = topic_embs.reshape(1, -1)

        # Cluster: merge turns with similar topics.
        n_turns = len(tids_in_session)
        if n_turns >= 3:
            sim_matrix = cosine_similarity(topic_embs)
            distance_matrix = 1.0 - sim_matrix
            np.fill_diagonal(distance_matrix, 0.0)

            # distance_threshold=0.55  ->  cosine_sim > 0.45  ->  merge
            try:
                clustering_dist_threshold = 0.55
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=clustering_dist_threshold,
                    metric='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
            except:
                # Fallback: every turn becomes its own cluster.
                labels = list(range(n_turns))
        elif n_turns == 2:
            sim = cosine_similarity(topic_embs)[0, 1] if topic_embs.shape[0] == 2 else 0.0
            labels = [0, 0] if sim > 0.55 else [0, 1]
        else:
            labels = [0]

        # Group by cluster label.
        cluster_groups = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_groups[int(label)].append((tids_in_session[i], topic_texts[i], topic_embs[i]))

        for label, members in cluster_groups.items():
            member_tids = [m[0] for m in members]
            member_texts = [m[1] for m in members]
            member_embs = np.array([m[2] for m in members])

            # Cluster text: use the longest topic phrase in the cluster.
            best_text = max(member_texts, key=len) if member_texts else ''

            topic_clusters.append({
                'id': f'topic_{cluster_idx}',
                'text': best_text,
                'turns': member_tids,
                'session': sk,
                'embedding': member_embs.mean(axis=0) if len(member_embs) > 0 else np.zeros(384, dtype=np.float32)
            })
            cluster_idx += 1

        # Per-session progress.
        if (si + 1) % 5 == 0 or si + 1 == total_sessions:
            print(f"  [TopicCluster] {sample_id}: {si+1}/{total_sessions} sessions clustered", file=sys.stderr)

    print(f"  [TopicCluster] {sample_id}: done -> {len(topic_clusters)} clusters across {total_sessions} sessions", file=sys.stderr)
    return topic_clusters


# ============================================================================
# Weight grid (3-D: bm25, turn, topic)
# ============================================================================
def generate_weight_grid_3d(step=0.10):
    """Enumerate every (alpha, beta, gamma) with step granularity and alpha+beta+gamma=1."""
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


# ============================================================================
# Cosine-similarity and min-max normalisation helpers
# ============================================================================
def _cosine_scores(query_emb, doc_embs):
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    return np.dot(d, q)


def _norm(scores_dict):
    """Min-max normalisation of a dictionary of scores into [0, 1]."""
    if not scores_dict:
        return scores_dict
    vals = list(scores_dict.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        return {k: 0.0 for k in scores_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores_dict.items()}


# ============================================================================
# Three-path hybrid recall + graph re-ranking (v2 update)
# ============================================================================
def graph_hybrid_recall_rerank_v2_update(query, meta, weights, embed_fn,
                                         sess_k=3, topic_k=5, bm25_n=100, top_k=10):
    """
    Three-path recall -> merge & de-duplicate -> multi-signal re-ranking.

    Path 1: BM25(query, turn)              top-N           (lexical backbone)
    Path 2: MiniLM(query, session_summary)  top-K sessions -> all turns
                                                     (semantic -> coarse topic)
    Path 3: MiniLM(query, topic_cluster)    top-K clusters -> only turns inside
                                                     the cluster (fine topic)

    weights = (w_bm25, w_turn, w_topic), sum=1.
    """
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn) = meta

    w_bm25, w_turn, w_topic = weights
    q_emb = embed_fn(query)
    q_toks = set(tokenize(query))

    candidate_tids = set()

    # ---- Path 1: BM25 top-N ----
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    for i in bm25_topn:
        candidate_tids.add(turn_all[i])

    # ---- Path 2: MiniLM -> top-K sessions -> all turns ----
    if len(sess_embs) > 0:
        s_scores = _cosine_scores(q_emb, sess_embs)
        sess_topk = np.argsort(s_scores)[::-1][:sess_k]
        for si in sess_topk:
            sk = sk_all[si]
            for i, tid in enumerate(turn_all):
                if turn_to_sess[tid] == sk:
                    candidate_tids.add(tid)

    # ---- Path 3: MiniLM -> top-K topic clusters -> only turns inside ----
    if len(topic_embs) > 0 and len(topic_to_turns) > 0:
        tc_scores = _cosine_scores(q_emb, topic_embs)
        tc_topk = np.argsort(tc_scores)[::-1][:topic_k]
        for tci in tc_topk:
            for tid in topic_to_turns.get(tci, []):
                candidate_tids.add(tid)

    if not candidate_tids:
        return []

    # ---- Re-rank with the three signals ----
    pool = list(candidate_tids)
    pool_indices = [turn_all.index(tid) for tid in pool]

    # BM25 score (min-max-normalised inside the candidate pool).
    bm25_pool = {tid: float(bm25_scores_all[idx]) for tid, idx in zip(pool, pool_indices)}
    bm25_norm = _norm(bm25_pool)

    # Turn embedding (min-max-normalised inside the candidate pool).
    cand_embs = turn_embs[pool_indices]
    t_cos = _cosine_scores(q_emb, cand_embs)
    turn_pool = {pool[i]: float(t_cos[i]) for i in range(len(pool))}
    turn_norm = _norm(turn_pool)

    # Topic-cluster score (broadcast to the turns inside each cluster).
    turn_topic_score = {}
    if len(topic_embs) > 0:
        tc_scores_all = _cosine_scores(q_emb, topic_embs)
        for tci in range(len(topic_to_turns)):
            score = float(tc_scores_all[tci])
            for tid in topic_to_turns.get(tci, []):
                # Keep the maximum topic score across the turn's clusters.
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
# Ablation: each path used on its own
# ============================================================================
def ablation_bm25_only(query, meta, bm25_n, top_k):
    """Ablation Path 1: BM25 recall only -> BM25 score -> top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn) = meta
    q_toks = set(tokenize(query))
    bm25_scores_all = bm25_turn.score_docs(q_toks)
    bm25_topn = np.argsort(bm25_scores_all)[::-1][:bm25_n]
    pool_tids = [turn_all[i] for i in bm25_topn]
    pool_scores = [float(bm25_scores_all[i]) for i in bm25_topn]
    ranked = sorted(zip(pool_tids, pool_scores), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


def ablation_sem_session_only(query, meta, embed_fn, sess_k, top_k):
    """Ablation Path 2: MiniLM->Session only -> turn-level cosine -> top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn) = meta
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
    """Ablation Path 3: MiniLM->Topic-Cluster only -> turn-level cosine -> top_k."""
    (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
     turn_all, turn_to_sess, turn_texts_ordered, bm25_turn) = meta
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


# ============================================================================
# BM25 implementation
# ============================================================================
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.lens = np.array([len(d) for d in docs])
        self.avgdl = np.mean(self.lens)
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
        """Score every document in the index (no truncation)."""
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
    from transformers import AutoModel, AutoTokenizer
    import torch
    print(f"[INFO] Downloading MiniLM from HuggingFace: {EMBED_MODEL}")
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
def rrf_retrieve_bm25_minilm_turn(query, turn_all, bm25_turn, q_emb, turn_embs,
                                  top_k=10, rrf_k=60):
    """BM25 + MiniLM embedding -> Reciprocal Rank Fusion (turn-level baseline)."""
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


def embed_retrieve(query_emb, doc_embs, top_k=10):
    scores = _cosine_scores(query_emb, doc_embs)
    idx = np.argsort(scores)[::-1][:top_k]
    return list(idx)


# ============================================================================
# Evaluation utilities
# ============================================================================
def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / max(len(gold), 1)


def mrr_score(ranked, gold):
    """MRR: 1 / rank_of_first_relevant (scans the entire ranked list)."""
    g = set(gold)
    for i, x in enumerate(ranked, 1):
        if x in g:
            return 1.0 / i
    return 0.0


def compute_f1(prediction, ground_truth):
    """Token-level F1."""
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
    """Exact match after normalisation."""
    def norm(s):
        return str(s).lower().strip().rstrip('.')
    return 1.0 if norm(prediction) == norm(ground_truth) else 0.0


def generate_answer_e2e(model, tokenizer, query, ranked_turns, turns_dict, max_retrieved=10):
    """
    Use the top-N retrieved turns as context and have Qwen generate the answer.

    Returns (answer_text, generation_time_seconds).
    """
    top_turns = ranked_turns[:max_retrieved]
    turn_texts = []
    for tid in top_turns:
        turn = turns_dict.get(tid, {})
        speaker = turn.get('speaker', '')
        text = turn.get('text', '')
        turn_texts.append(f"{speaker}: {text}" if speaker and text else (text or ' '))

    context = '\n'.join(turn_texts)

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

    answer = raw.split('\n')[0]
    return answer, gen_time


# ============================================================================
# Main experiment
# ============================================================================
CAT_NAMES = {1: 'single-hop', 2: 'temporal', 3: 'inference',
             4: 'multi-hop', 5: 'adversarial'}

TOP_K = 10  # Unified top_k (changed from 20 to 10).


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='long-term-memory/locomo10.json')
    parser.add_argument('--qwen_path', type=str, default=None,
                        help='Path to Qwen2.5-7B-Instruct (only needed for the first run that generates turn topics).')
    parser.add_argument('--turn_topic_cache', type=str, default='turn_topics_cache.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_convs', type=int, default=None)
    parser.add_argument('--max_qa_per_conv', type=int, default=None)
    parser.add_argument('--max_qa_total', type=int, default=None)
    parser.add_argument('--weight_search', action='store_true')
    parser.add_argument('--grid_step', type=float, default=0.10)
    parser.add_argument('--sem_session_k', type=int, default=3,
                        help='Number of sessions retrieved by the MiniLM session path (default: 3).')
    parser.add_argument('--sem_topic_k', type=int, default=5,
                        help='Number of topic clusters retrieved by the MiniLM topic path (default: 5).')
    parser.add_argument('--sem_bm25_n', type=int, default=100,
                        help='BM25 path top-N (default: 100).')
    parser.add_argument('--end_to_end', action='store_true',
                        help='Run end-to-end answer generation + F1/EM (requires --qwen_path).')
    parser.add_argument('--max_retrieved', type=int, default=10,
                        help='Number of retrieved turns fed into the E2E generator (default: 10).')
    args = parser.parse_args()

    # ===== Load the dataset =====
    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    if args.max_convs:
        data = data[:args.max_convs]
        print(f"[INFO] Small-sample mode: {len(data)} conversations")

    if args.max_qa_total:
        total_available = sum(len(item['qa']) for item in data)
        per_conv = max(1, args.max_qa_total // len(data))
        print(f"[INFO] Total QA limit: {args.max_qa_total} "
              f"(sampling ~{per_conv} per conversation, available: {total_available})")

    # ===== Phase 0: load / generate turn-topic cache (Qwen) =====
    if args.qwen_path:
        turn_topics_cache = load_or_generate_turn_topics(
            data, args.qwen_path, args.turn_topic_cache
        )
    else:
        if not os.path.exists(args.turn_topic_cache):
            print(f"[ERROR] Turn topic cache not found: {args.turn_topic_cache}")
            print("  First run: python graph_vs_baselines_LoCoMo.py "
                  "--qwen_path /path/to/Qwen2.5-7B-Instruct")
            return
        print(f"[INFO] Loading cached turn topics from {args.turn_topic_cache}")
        with open(args.turn_topic_cache, 'r', encoding='utf-8') as f:
            turn_topics_cache = json.load(f)

    # ===== Load MiniLM =====
    embed = load_embedder()

    # ===== Phase 1: build the graph + pre-compute embeddings + collect QA =====
    # meta = (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
    #         turn_all, turn_to_sess, turn_texts_ordered, bm25_turn)
    all_samples = []
    conv_meta = []
    build_t0 = time.time()

    for ci, item in enumerate(data):
        print(f"[Conv {ci + 1}/{len(data)}] {item['sample_id']}  building graph + embedding...")
        graph = build_graph_v2_update(item, turn_topics_cache, embed)

        sk_all = graph['sk_all']
        sessions = graph['sessions']
        turns = graph['turns']
        topic_clusters = graph['topic_clusters']
        turn_all = graph['turn_all']
        turn_to_sess = graph['turn_to_sess']

        # Session embeddings.
        sess_texts = [sessions[sk]['text'] for sk in sk_all]
        sess_embs = embed(sess_texts)

        # Turn embeddings (in `turn_all` order).
        turn_texts_ordered = []
        for tid in turn_all:
            turn = turns[tid]
            txt = turn['text'].strip()
            turn_texts_ordered.append(
                f"{turn['speaker']}: {txt}" if turn['speaker'] and txt else (txt or ' ')
            )
        print(f"           Embedding {len(turn_texts_ordered)} turns...", file=sys.stderr)
        turn_embs = embed(turn_texts_ordered)

        # Turn BM25.
        turn_tokens = [tokenize(t) for t in turn_texts_ordered]
        bm25_turn = BM25(turn_tokens)

        # Topic-cluster embeddings (already computed in the graph).
        topic_texts = [tc['text'] for tc in topic_clusters]
        print(f"           Embedding {len(topic_texts)} topic clusters...", file=sys.stderr)
        if topic_texts:
            topic_embs_raw = embed(topic_texts)
            if topic_embs_raw.ndim == 1:
                topic_embs_raw = topic_embs_raw.reshape(1, -1)
        else:
            topic_embs_raw = np.zeros((0, 384), dtype=np.float32)

        # topic_to_turns: cluster index -> list of turn IDs.
        topic_to_turns = {}
        for tci, tc in enumerate(topic_clusters):
            topic_to_turns[tci] = tc['turns']

        print(f"           L3 sessions: {len(sessions)}, L2 turns: {len(turns)}, "
              f"L1 topic clusters: {len(topic_clusters)}")

        conv_meta.append((sk_all, sess_embs, turn_embs, topic_embs_raw, topic_to_turns,
                          turn_all, turn_to_sess, turn_texts_ordered, bm25_turn))

        # ---- QA collection (gold = turn IDs from `evidence`) ----
        qa_list = item['qa']
        if args.max_qa_per_conv:
            qa_list = qa_list[:args.max_qa_per_conv]
        if args.max_qa_total:
            per_conv = max(1, args.max_qa_total // len(data))
            qa_list = qa_list[:per_conv]

        for qa in qa_list:
            query = qa.get('question', '')
            cat = qa.get('category', 0)
            evidence = qa.get('evidence', [])
            if not evidence:
                continue

            # Gold = the turn IDs that appear in the `evidence` field.
            gold = [evi for evi in evidence if evi in turns]
            if not gold:
                continue

            all_samples.append((ci, query, gold, cat, turns, qa.get('answer', '')))

    total_qa = len(all_samples)
    build_time = time.time() - build_t0
    print(f"\n[INFO] Collected {total_qa} valid QA pairs across {len(data)} conversations "
          f"(pre-build time: {build_time:.1f}s)\n")

    if total_qa == 0:
        print("[ERROR] No QA pairs found! Exiting.")
        return

    # ===== Phase 2: grid-search or fixed weights =====
    grid_results = None
    weights = (0.40, 0.30, 0.30)  # Default: w_bm25, w_turn, w_topic.
    retrieve_fn = lambda q, m, w: graph_hybrid_recall_rerank_v2_update(
        q, m, w, embed, args.sem_session_k, args.sem_topic_k, args.sem_bm25_n, TOP_K)
    mode_name = 'Graph_TurnTopicRecall'

    if args.weight_search:
        weight_grid = generate_weight_grid_3d(args.grid_step)
        print(f"{'='*80}")
        print(f"WEIGHT GRID SEARCH: {len(weight_grid)} combinations "
              f"(step={args.grid_step}, 3D: bm25+turn+topic)")
        print(f"{'='*80}")

        all_grid_results = []
        best = {'weights': None, 'r10': 0.0}
        report_every = max(1, len(weight_grid) // 20)

        for gi, w in enumerate(weight_grid):
            r1_vals, r3_vals, r5_vals, r10_vals, mrr_vals = [], [], [], [], []
            for ci, query, gold, cat, _turns, _gt_answer in all_samples:
                meta = conv_meta[ci]
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
                marker = ' *' if is_new_best else '  '
                print(f"  [{gi+1:>4}/{len(weight_grid)} {pct:5.1f}%]{marker} "
                      f"R@10={avg_r10:.4f} (best={best['r10']:.4f})")

        weights = best['weights']
        all_grid_results.sort(key=lambda x: x['R@10'], reverse=True)
        grid_results = all_grid_results  # Save every combination for later analysis.

        print(f"\n[Grid Search] BEST -> R@10={best['r10']:.4f}  "
              f"R@5={best['r5']:.4f}  MRR={best['mrr']:.4f}")
        label_names = ['bm25', 'turn', 'topic']
        print(f"  Weights: {dict(zip(label_names, [f'{v:.2f}' for v in weights]))}")
        print(f"\n[Grid Search] All {len(all_grid_results)} combinations (sorted by R@10):")
        print(f"  {'Rank':<5} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}   bm25    turn    topic")
        for rank, r in enumerate(all_grid_results, 1):
            ws = r['weights']
            marker = ' *' if (ws['bm25'] == round(weights[0], 2) and
                              ws['turn'] == round(weights[1], 2) and
                              ws['topic'] == round(weights[2], 2)) else '  '
            print(f"  {rank:<5} {r['R@1']:>8.4f} {r['R@3']:>8.4f} {r['R@5']:>8.4f} "
                  f"{r['R@10']:>8.4f} {r['MRR']:>8.4f}   "
                  f"{ws['bm25']:.2f}   {ws['turn']:.2f}   {ws['topic']:.2f}{marker}")
    else:
        print(f"[INFO] TurnTopicRecall default weights: "
              f"bm25={weights[0]:.2f} turn={weights[1]:.2f} topic={weights[2]:.2f}")

    # ===== Phase 3: retrieval evaluation (turn-level, MRR@10) =====
    ablation_methods = ['Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    methods = [mode_name, 'BM25+MiniLM_RRF', 'MiniLM', 'BM25'] + ablation_methods
    overall = {m: defaultdict(list) for m in methods}
    per_cat = {m: defaultdict(lambda: defaultdict(list)) for m in methods}
    retrieval_times = defaultdict(list)  # per-method per-query retrieval time

    # Cache per-query retrieval results for the E2E phase.
    all_ranked = []

    for eval_idx, (ci, query, gold, cat, turns, gt_answer) in enumerate(all_samples):
        meta = conv_meta[ci]
        (sk_all, sess_embs, turn_embs, topic_embs, topic_to_turns,
         turn_all, turn_to_sess, turn_texts_ordered, bm25_turn) = meta

        # Graph (turn-topic recall + re-rank) -> top-10.
        t0 = time.time()
        g_ranked = retrieve_fn(query, meta, weights)
        retrieval_times[mode_name].append(time.time() - t0)

        q_emb = embed(query)
        q_toks = set(tokenize(query))

        # BM25 + MiniLM RRF (turn-level baseline) -> top-10.
        t0 = time.time()
        rrf_ranked = rrf_retrieve_bm25_minilm_turn(
            query, turn_all, bm25_turn, q_emb, turn_embs, top_k=TOP_K, rrf_k=60)
        retrieval_times['BM25+MiniLM_RRF'].append(time.time() - t0)

        # MiniLM (turn level) -> top-10.
        t0 = time.time()
        m_idx = embed_retrieve(q_emb, turn_embs, top_k=TOP_K)
        m_ranked = [turn_all[i] for i in m_idx]
        retrieval_times['MiniLM'].append(time.time() - t0)

        # BM25 (turn level) -> top-10.
        t0 = time.time()
        b_idx = bm25_turn.search(q_toks, top_k=TOP_K)
        b_ranked = [turn_all[i] for i in b_idx]
        retrieval_times['BM25'].append(time.time() - t0)

        # ---- Ablation: each recall path on its own ----
        t0 = time.time()
        abl_bm25_ranked = ablation_bm25_only(query, meta, args.sem_bm25_n, TOP_K)
        retrieval_times['Abl_BM25'].append(time.time() - t0)

        t0 = time.time()
        abl_sess_ranked = ablation_sem_session_only(query, meta, embed, args.sem_session_k, TOP_K)
        retrieval_times['Abl_SemSession'].append(time.time() - t0)

        t0 = time.time()
        abl_topic_ranked = ablation_sem_topic_only(query, meta, embed, args.sem_topic_k, TOP_K)
        retrieval_times['Abl_SemTopic'].append(time.time() - t0)

        # Cache ranked results for the E2E phase.
        all_ranked.append({
            'query': query, 'gold': gold, 'cat': cat,
            'turns': turns, 'gt_answer': gt_answer,
            mode_name: g_ranked,
            'BM25+MiniLM_RRF': rrf_ranked,
            'MiniLM': m_ranked,
            'BM25': b_ranked,
            'Abl_BM25': abl_bm25_ranked,
            'Abl_SemSession': abl_sess_ranked,
            'Abl_SemTopic': abl_topic_ranked,
        })

        for method, ranked in [(mode_name, g_ranked), ('BM25+MiniLM_RRF', rrf_ranked),
                                ('MiniLM', m_ranked), ('BM25', b_ranked),
                                ('Abl_BM25', abl_bm25_ranked),
                                ('Abl_SemSession', abl_sess_ranked),
                                ('Abl_SemTopic', abl_topic_ranked)]:
            for k in [1, 3, 5, 10]:
                overall[method][f'R@{k}'].append(recall_at_k(ranked, gold, k))
                per_cat[method][cat][f'R@{k}'].append(recall_at_k(ranked, gold, k))
            # MRR scanned over the full ranked list (~ MRR@10 since ranked length is 10).
            overall[method]['MRR'].append(mrr_score(ranked, gold))
            per_cat[method][cat]['MRR'].append(mrr_score(ranked, gold))

        # Progress bar.
        if (eval_idx + 1) % 50 == 0 or eval_idx + 1 == total_qa:
            pct = 100 * (eval_idx + 1) / total_qa
            print(f"  [Retrieval Eval] {eval_idx + 1}/{total_qa} ({pct:.1f}%)", flush=True)


    # ===== Phase 4: E2E evaluation (R@10 -> Qwen answer -> F1/EM) =====
    baseline_e2e_methods = ['BM25+MiniLM_RRF', 'MiniLM', 'BM25']
    e2e_methods = [mode_name] + baseline_e2e_methods + ablation_methods
    e2e_all = {}
    if args.end_to_end and args.qwen_path:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n{'='*80}")
        print("END-TO-END ANSWER GENERATION + F1/EM EVALUATION")
        print(f"{'='*80}")

        # Load the Qwen model.
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
        e2e_total_time = {m: [] for m in e2e_methods}  # retrieval + generation

        e2e_total = len(all_ranked)
        for e2e_idx, r in enumerate(all_ranked):
            query = r['query']
            turns_dict = r['turns']
            gt_answer = r['gt_answer']

            if not gt_answer:
                continue

            for em_name in e2e_methods:
                ranked = r[em_name]
                pred_answer, gen_t = generate_answer_e2e(
                    e2e_model, e2e_tokenizer, query, ranked, turns_dict,
                    max_retrieved=args.max_retrieved
                )
                ret_t = retrieval_times[em_name][e2e_idx]
                f1, _recall = compute_f1(pred_answer, gt_answer)
                em = compute_exact_match(pred_answer, gt_answer)
                e2e_f1[em_name].append(f1)
                e2e_em[em_name].append(em)
                e2e_gen_time[em_name].append(gen_t)
                e2e_total_time[em_name].append(ret_t + gen_t)

            # Progress bar.
            if (e2e_idx + 1) % 10 == 0 or e2e_idx + 1 == e2e_total:
                pct = 100 * (e2e_idx + 1) / e2e_total
                parts = []
                for em_name in e2e_methods:
                    f = np.mean(e2e_f1[em_name]) if e2e_f1[em_name] else 0.0
                    parts.append(f"{em_name}:F1={f:.4f}")
                print(f"  [E2E] {e2e_idx + 1}/{e2e_total} ({pct:.1f}%) | " + " | ".join(parts),
                      flush=True)

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

    # ===== Phase 5: print and save the results =====
    print("\n" + "=" * 90)
    suffix = " (grid-best)" if args.weight_search else ""
    all_display_methods = [f"{mode_name}{suffix}", 'BM25+MiniLM_RRF', 'MiniLM', 'BM25',
                           'Abl_BM25', 'Abl_SemSession', 'Abl_SemTopic']
    print(f"RESULTS ({total_qa} QA pairs, {len(data)} conversations) [MRR@{TOP_K}]")
    print("=" * 90)
    metrics = ['R@1', 'R@3', 'R@5', 'R@10', 'MRR']
    header = f"{'Metric':<8}" + "".join(f"{m:>22}" for m in all_display_methods)
    print(header)
    print("-" * 90)
    for met in metrics:
        vals = [np.mean(overall[m][met]) for m in methods]
        print(f"{met:<8}" + "".join(f"{v:>22.4f}" for v in vals))

    # E2E summary (main method + baselines + ablations).
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
        # Total pipeline time (retrieval + generation).
        row = f"{'ret+gen(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_total_pipeline_time_s', 0.0)
            row += f"{val:>22.2f}"
        print(row)
        # Pure generation time.
        row = f"{'gen_only(s)':<14}"
        for em_name in e2e_display:
            val = e2e_all.get(em_name, {}).get('avg_generation_time_s', 0.0)
            row += f"{val:>22.2f}"
        print(row)

    # ===== Retrieval-time statistics =====
    print("\n" + "-" * 90)
    print("AVERAGE RETRIEVAL TIME PER QUERY (ms)")
    print(f"{'Method':<22}" + f"{'Avg Time':>10}")
    print("-" * 35)
    for m in methods:
        times = retrieval_times.get(m, [])
        avg_ms = np.mean(times) * 1000 if times else 0
        print(f"{m:<22} {avg_ms:>9.1f}ms")

    print("\n" + "-" * 90)
    print("PER-CATEGORY (R@5)")
    print(f"{'Category':<16}" + "".join(f"{m:>22}" for m in all_display_methods))
    print("-" * 90)
    for cid in sorted(CAT_NAMES):
        row = f"{CAT_NAMES[cid]:<16}"
        for m in methods:
            vals = per_cat[m][cid].get('R@5', [])
            row += f"{np.mean(vals):>22.4f}" if vals else f"{'N/A':>22}"
        print(row)

    # ===== Save the result =====
    out = {
        'version': 'graph_turn_topic_recall',
        'method': (f'BM25(top{args.sem_bm25_n}) + MiniLM->sess(top{args.sem_session_k}) '
                   f'+ MiniLM->topic_cluster(top{args.sem_topic_k})'),
        'top_k': TOP_K,
        'sem_session_k': args.sem_session_k,
        'sem_topic_k': args.sem_topic_k,
        'sem_bm25_n': args.sem_bm25_n,
        'total_qa': total_qa,
        'num_convs': len(data),
        'weight_search': args.weight_search,
        'weights': {k: float(v) for k, v in zip(['bm25', 'turn', 'topic'], weights)},
        'pre_build_time_s': round(build_time, 1),
        'avg_retrieval_time_ms': {},
        'overall': {
            m: {k: float(np.mean(v)) if v else 0.0
                for k, v in ov.items()}
            for m, ov in overall.items()
        },
        'ablation': {
            'Abl_BM25': 'Path 1 only: BM25 query->turn recall.',
            'Abl_SemSession': 'Path 2 only: MiniLM(query, session_summary) recall.',
            'Abl_SemTopic': 'Path 3 only: MiniLM(query, topic_cluster) recall.',
        },
        'per_category_r5': {},
        'per_category_r10': {}
    }
    for m in methods:
        times = retrieval_times.get(m, [])
        out['avg_retrieval_time_ms'][m] = float(np.mean(times) * 1000) if times else 0.0

    for cid in sorted(CAT_NAMES):
        out['per_category_r5'][CAT_NAMES[cid]] = {}
        out['per_category_r10'][CAT_NAMES[cid]] = {}
        for m in methods:
            vals5 = per_cat[m][cid].get('R@5', [])
            vals10 = per_cat[m][cid].get('R@10', [])
            out['per_category_r5'][CAT_NAMES[cid]][m] = float(np.mean(vals5)) if vals5 else None
            out['per_category_r10'][CAT_NAMES[cid]][m] = float(np.mean(vals10)) if vals10 else None

    if e2e_all:
        out['e2e_all'] = e2e_all

    if grid_results is not None:
        out['grid_search'] = {
            'step': args.grid_step,
            'num_combos_tested': len(grid_results),
            'all_results': grid_results
        }

    outpath = 'graph_turn_topic_recall_result.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()