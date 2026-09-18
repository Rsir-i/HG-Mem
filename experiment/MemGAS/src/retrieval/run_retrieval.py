import os
import time
import multiprocessing as mp
from functools import partial
from collections import Counter
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from eval_utils import evaluate_retrieval
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, src_path)

from src.construct.construct_emb import emb_rawdata
from src.construct.construct_asso import construct_asso, construct_asso_turn

# ── F1 tokenization (consistent with graph_vs_baselines_v3) ──
_STOPS = {'the','a','an','is','are','was','were','in','on','at','to','for',
          'of','with','and','or','by','from','it','its','i','you','he','she',
          'we','they','my','your','his','her','our','their','me','him','us',
          'them','that','this','these','those','be','been','being','have','has',
          'had','do','does','did','will','would','shall','should','may','might',
          'can','could','not','no','but','if','so','as','than','then','just',
          'also','very','too','only','all','some','any','each','every','both',
          'few','more','most','other','own','same','such','up','down','out',
          'about','into','over','after','before','between','through','during',
          'above','below','under','again','further','here','there','when',
          'where','why','how','which','who','whom','what'}

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    _HAS_STEMMER = True
except ImportError:
    _stemmer = None
    _HAS_STEMMER = False


def tokenize(text):
    """Tokenize in the same way as graph_vs_baselines_v3."""
    t = text.lower()
    for ch in '?!.,;:\"\'()[]{}-\n\r':
        t = t.replace(ch, ' ')
    words = [w for w in t.split() if w not in _STOPS and len(w) > 1]
    if _HAS_STEMMER and _stemmer:
        words = [_stemmer.stem(w) for w in words]
    return words


def compute_f1(prediction, ground_truth) -> float:
    """Token-level F1 (consistent with graph_vs_baselines_v3: set intersection
    + stemming + stopword removal)."""
    pred_tokens = set(tokenize(str(prediction)))
    gt_tokens = set(tokenize(str(ground_truth)))
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def run_ppr(g,reset_prob, damping):
    reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
    pagerank_scores = g.personalized_pagerank(
        # vertices=vertices,
        damping=damping,
        directed=False,
        # weights=g.es["weight"],
        reset=reset_prob,
        implementation='prpack'
    )
    return pagerank_scores
    # return torch.tensor(pagerank_scores).argsort(descending=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # basic parameters
    parser.add_argument('--retriever', type=str, required=True,)
    parser.add_argument('--method', type=str, required=True)

    parser.add_argument('--num_seednodes', type=int, default=15)
    parser.add_argument('--mem_threshold', type=int, default=30, help="Control the candidate set size.")
    parser.add_argument('--n_components', type=int, default=2, help="Control the number of GMM clustering categories.")
    parser.add_argument('--damping', type=float, default=0.1, help="")
    parser.add_argument('--temp', type=float, default=0.1, help="")
    return parser.parse_args()



def multi_granularity_routing(args, query_emb, granular_embeddings):
    entropies = []
    for emb in granular_embeddings:
        similarity = (query_emb @ emb.T).squeeze()
        prob_dist = F.softmax(similarity / args.temp, dim=0)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
        entropies.append(entropy)
    entropies = torch.tensor(entropies)
    # entropies = entropies / torch.sum(entropies)
    # soft_router_weights = (1 - entropies) / sum(1 - entropies) # Shape: (len(granular_embeddings),)

    soft_router_weights = 1 - entropies
    soft_router_weights /= soft_router_weights.sum()
    return soft_router_weights

def main(args):
    emb_path = f'../../data/process_embs/{args.dataset}-{args.retriever}-emb.pt'
    if os.path.exists(emb_path):
        all_emb = torch.load(emb_path, weights_only=False)
    else:
        all_emb = emb_rawdata(args.dataset, args.retriever)
    
    in_data = json.load(open(f'../../data/process_data/{args.dataset}.json'))
    
    if args.method in ('memgas', 'memgas_session2turn'):
        graph_path = f"../../graph_cache/graph-{args.dataset}-{args.retriever}-{args.mem_threshold}-{args.n_components}.pt"
        if os.path.exists(graph_path):
            covid2graph = torch.load(graph_path, weights_only=False)
        else:
            covid2graph = construct_asso(args)
    if args.method == 'memgas_turn':
        graph_path = f"../../graph_cache/graph-turn-{args.dataset}-{args.retriever}-{args.mem_threshold}-{args.n_components}.pt"
        if os.path.exists(graph_path):
            covid2graph_turn = torch.load(graph_path, weights_only=False)
        else:
            covid2graph_turn = construct_asso_turn(args)
    results = []
    # ── Progress tracking ──
    total_qa = sum(len(entry['qa']) for entry in in_data)
    qa_done = 0
    gen_latencies = []  # latency of the questions that ran generation
    gen_methods = ('turn_retrieval', 'memgas', 'memgas_turn', 'memgas_session2turn')
    if args.method in gen_methods:
        print(f"[progress] {total_qa} questions in total, starting retrieval + generation evaluation...")
    for entry, emb in zip(in_data,all_emb):
        assert entry['conversation_id'] == emb['conversation_id']

        #################### turn emb mean (per-session, for session-level methods)
        turn_num_each_session = [len(sess) for sess in entry['sessions']]
        turn_embeddings = []
        start_idx = 0
        for num_turns in turn_num_each_session:
            if num_turns == 0:
                turn_mean_emb = torch.zeros(emb['turns'].size(1))
            else:
                session_turn_embs = emb['turns'][start_idx:start_idx + num_turns]
                turn_mean_emb = session_turn_embs.mean(dim=0)
            turn_embeddings.append(turn_mean_emb)
            start_idx += num_turns
        turn_embeddings = torch.stack(turn_embeddings)
        ####################

        #################### turn corpus (individual merged turns, for turn_retrieval)
        turn_corpus_ids = []
        turn_corpus_embs = []
        turn_corpus_texts = []
        t_idx = 0
        for sess_id, session in zip(entry['sessions_ids'], entry['sessions']):
            num_merged_turns = len(session)
            for j in range(num_merged_turns):
                turn_corpus_ids.append(f"{sess_id}-turn_{j+1}")
                turn_corpus_embs.append(emb['turns'][t_idx])
                turn_corpus_texts.append(session[j])
                t_idx += 1
        turn_corpus_embs = torch.stack(turn_corpus_embs) if turn_corpus_embs else torch.empty(0, emb['turns'].size(1))
        ####################

        #################### turn-level multi-granularity embeddings (memgas_turn)
        if args.method == 'memgas_turn':
            turn_session_embs = []
            turn_summary_embs = []
            turn_keyword_embs = []
            t_idx = 0
            for sess_idx, session in enumerate(entry['sessions']):
                for j in range(len(session)):
                    turn_session_embs.append(emb['sessions'][sess_idx])
                    turn_summary_embs.append(emb['summarys'][sess_idx])
                    turn_keyword_embs.append(emb['keywords'][sess_idx])
                    t_idx += 1
            turn_session_embs = torch.stack(turn_session_embs)
            turn_summary_embs = torch.stack(turn_summary_embs)
            turn_keyword_embs = torch.stack(turn_keyword_embs)
        ####################

        for qa_one, q_emb in zip(entry['qa'],emb['questions']):
            if args.dataset != "LongMTBench+":
                correct_docs = list(set( [ids for ids in qa_one['answer_session_ids']] ))

            if args.method == 'session_level':
                scores = (q_emb @ emb['sessions'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'keyword_level':
                scores = (q_emb @ emb['keywords'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'summary_level':
                scores = (q_emb @ emb['summarys'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'hybrid_level':
                scores = (q_emb @ emb['hybrid'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'turn_level':
                scores = (q_emb @ turn_embeddings.T).squeeze()
                rankings = scores.argsort(descending=True)

            elif args.method == 'turn_retrieval':
                # retrieve over individual turns directly and evaluate with answer_turn_ids
                scores = (q_emb @ turn_corpus_embs.T).squeeze()
                rankings = scores.argsort(descending=True)

            elif args.method == 'lite':
                # Static weighted fusion: 0.35*session + 0.15*turn + 0.30*summary + 0.20*keyword
                weights = [0.35, 0.15, 0.30, 0.20]
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'], emb['keywords']]
                fused_scores = sum(w * (q_emb @ e.T).squeeze() for w, e in zip(weights, emb_list))
                rankings = fused_scores.argsort(descending=True)

            elif args.method == 'memgas':
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'],emb['keywords']]
                soft_router_weights = multi_granularity_routing(args, q_emb, emb_list)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(emb['sessions'].size(0)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])

                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()

                topk_values, _ = torch.topk(scores, args.num_seednodes)
                scores[scores < topk_values[-1]] = 0
                scores = run_ppr(covid2graph[entry['conversation_id']], scores, args.damping)

                scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]
                rankings = torch.tensor(scores).argsort(descending=True)

            elif args.method == 'memgas_session2turn':
                # Stage 1: session-level multi-granularity PPR retrieval, identical to memgas
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'], emb['keywords']]
                soft_router_weights = multi_granularity_routing(args, q_emb, emb_list)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(emb['sessions'].size(0)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])

                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()

                topk_values, _ = torch.topk(scores, args.num_seednodes)
                scores[scores < topk_values[-1]] = 0
                scores = run_ppr(covid2graph[entry['conversation_id']], scores, args.damping)

                # Aggregate: every 4 nodes correspond to one session
                session_scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]
                session_ranking = torch.tensor(session_scores).argsort(descending=True)

                # Stage 2: expand sessions into turns; each turn inherits its session's
                # PPR score and ties are broken by chronological order
                turn_to_session = []
                for sess_idx, num_turns in enumerate(turn_num_each_session):
                    turn_to_session.extend([sess_idx] * num_turns)

                # A turn's score = the PPR score of its session; ties keep the natural
                # chronological order
                turn_scored = []
                for ti, si in enumerate(turn_to_session):
                    turn_scored.append((session_scores[si], ti))
                # Sort by session score descending, ties by turn chronological order (ti ascending)
                turn_scored.sort(key=lambda x: (-x[0], x[1]))
                rankings = torch.tensor([ti for _, ti in turn_scored])

            elif args.method == 'memgas_turn':
                # Turn-level multi-granularity: each turn has 4 granularity embeddings
                emb_list = [turn_session_embs, turn_corpus_embs, turn_summary_embs, turn_keyword_embs]
                soft_router_weights = multi_granularity_routing(args, q_emb, emb_list)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(len(turn_corpus_ids)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])

                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()

                topk_values, _ = torch.topk(scores, args.num_seednodes)
                scores[scores < topk_values[-1]] = 0
                scores = run_ppr(covid2graph_turn[entry['conversation_id']], scores, args.damping)

                # Aggregate: every 4 nodes correspond to one turn
                scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]
                rankings = torch.tensor(scores).argsort(descending=True)

            # ── Build the result record ──
            if args.method in ('turn_retrieval', 'memgas_turn', 'memgas_session2turn'):
                cur_results = {
                    "conversation_id": entry['conversation_id'],
                    'question_type': qa_one['question_type'],
                    'question': qa_one['question'],
                    'answer': qa_one['answer'],
                    'question_date': qa_one['question_date'],
                    'retrieval_results': {
                        'ranked_items': [
                            {'corpus_id': turn_corpus_ids[rid]}
                            for rid in rankings
                        ],
                        'metrics': {
                            'session': {},
                            'turn': {}
                        }
                    }
                }
            else:
                cur_results = {
                    "conversation_id": entry['conversation_id'],
                    'question_type': qa_one['question_type'],
                    'question': qa_one['question'],
                    'answer': qa_one['answer'],
                    'question_date': qa_one['question_date'],
                    'retrieval_results': {
                        'ranked_items': [
                            {
                                'corpus_id': entry['sessions_ids'][rid],
                                'timestamp': entry['sessions_dates'][rid],
                            }
                            for rid in rankings
                        ],
                        'metrics': {
                            'session': {},
                            'turn': {}
                        }
                    }
                }

            if args.dataset != "LongMTBench+":
                if args.method in ('turn_retrieval', 'memgas_turn', 'memgas_session2turn'):
                    # turn-level evaluation: correct_docs uses answer_turn_ids, the corpus uses turn_corpus_ids
                    correct_turn_docs = list(set(qa_one.get('answer_turn_ids', [])))
                    if correct_turn_docs:
                        for k in [1, 3, 5, 10, 30, 50]:
                            recall, ndcg_val, mr = evaluate_retrieval(rankings, correct_turn_docs, turn_corpus_ids, k=k)
                            cur_results['retrieval_results']['metrics']['turn'].update({
                                'recall@{}'.format(k): recall,
                                'ndcg@{}'.format(k): ndcg_val,
                                'mrr@{}'.format(k): mr,
                            })

                    # ── End-to-end generation evaluation: answer from the top-10 retrieved turns + query ──
                    try:
                        from local_qwen import get_local_llm
                    except ImportError:
                        from src.local_qwen import get_local_llm

                    top_k_gen = min(10, len(rankings))
                    context_parts = []
                    for rank_idx in rankings[:top_k_gen]:
                        context_parts.append(turn_corpus_texts[int(rank_idx)])
                    context = "\n\n---\n\n".join(context_parts)

                    gen_prompt = (
                        "Based on the following conversation context, answer the question briefly and accurately.\n\n"
                        f"Conversation:\n{context}\n\n"
                        f"Question: {qa_one['question']}\n\n"
                        "Answer:"
                    )

                    llm = get_local_llm()
                    t_start = time.time()
                    generated = llm.generate(gen_prompt, max_tokens=200, temperature=0.0)
                    latency_ms = (time.time() - t_start) * 1000

                    f1 = compute_f1(generated, qa_one['answer'])

                    cur_results['generated_answer'] = generated
                    cur_results['f1_score'] = f1
                    cur_results['latency_ms'] = round(latency_ms, 2)

                    # ── Progress ──
                    gen_latencies.append(latency_ms)
                    qa_done += 1
                    if qa_done % 10 == 0 or qa_done == total_qa:
                        avg_lat = np.mean(gen_latencies)
                        print(f"  [{qa_done}/{total_qa}] avg latency: {avg_lat:.0f} ms/q | last: {latency_ms:.0f} ms", flush=True)

                elif args.method == 'memgas':
                    # session-level evaluation
                    for k in [1, 3, 5, 10, 30, 50]:
                        recall, ndcg_val, mrr_score = evaluate_retrieval(rankings, correct_docs, entry['sessions_ids'], k=k)
                        cur_results['retrieval_results']['metrics']['session'].update({
                            'recall@{}'.format(k): recall,
                            'ndcg@{}'.format(k): ndcg_val,
                            'mrr@{}'.format(k): mrr_score,
                        })

                    # ── End-to-end generation evaluation: concatenate all turns of the
                    #    top-5 retrieved sessions + query ──
                    try:
                        from local_qwen import get_local_llm
                    except ImportError:
                        from src.local_qwen import get_local_llm

                    top_k_gen = min(5, len(rankings))
                    context_parts = []
                    for rank_idx in rankings[:top_k_gen]:
                        sid = int(rank_idx)
                        session_turns = entry['sessions'][sid]
                        context_parts.append("\n\n".join(session_turns))
                    context = "\n\n========\n\n".join(context_parts)

                    gen_prompt = (
                        "Based on the following conversation sessions, answer the question briefly and accurately.\n\n"
                        f"Conversation:\n{context}\n\n"
                        f"Question: {qa_one['question']}\n\n"
                        "Answer:"
                    )

                    llm = get_local_llm()
                    t_start = time.time()
                    generated = llm.generate(gen_prompt, max_tokens=200, temperature=0.0)
                    latency_ms = (time.time() - t_start) * 1000

                    f1 = compute_f1(generated, qa_one['answer'])

                    cur_results['generated_answer'] = generated
                    cur_results['f1_score'] = f1
                    cur_results['latency_ms'] = round(latency_ms, 2)

                    # ── Progress ──
                    gen_latencies.append(latency_ms)
                    qa_done += 1
                    if qa_done % 10 == 0 or qa_done == total_qa:
                        avg_lat = np.mean(gen_latencies)
                        print(f"  [{qa_done}/{total_qa}] avg latency: {avg_lat:.0f} ms/q | last: {latency_ms:.0f} ms", flush=True)

                else:
                    for k in [1, 3, 5, 10, 30, 50]:
                        recall, ndcg_val, mrr_score = evaluate_retrieval(rankings, correct_docs, entry['sessions_ids'], k=k)
                        cur_results['retrieval_results']['metrics']['session'].update({
                            'recall@{}'.format(k): recall,
                            'ndcg@{}'.format(k): ndcg_val,
                            'mrr@{}'.format(k): mrr_score,
                        })
            results.append(cur_results)
            
    if args.dataset != "LongMTBench+":
        # Compute and print retrieval metrics: R@k, MRR
        print("\n" + "=" * 80)
        print(f"Retrieval Results: {args.dataset} | {args.retriever} | {args.method}")
        print("=" * 80)
        metric_key = 'turn' if args.method in ('turn_retrieval', 'memgas_turn', 'memgas_session2turn') else 'session'
        if results[0]['retrieval_results']['metrics'][metric_key]:
            metric_names = [k for k in results[0]['retrieval_results']['metrics'][metric_key]]
            for k_name in metric_names:
                vals = [x['retrieval_results']['metrics'][metric_key][k_name]
                        for x in results
                        if '_abs' not in str(x['conversation_id'])
                        and k_name in x['retrieval_results']['metrics'][metric_key]]
                if vals:
                    k_result = np.mean(vals)
                    print(f"  {k_name}: {k_result*100:.2f}%")

        # ── End-to-end generation metrics (turn_retrieval / memgas / memgas_turn) ──
        if args.method in ('turn_retrieval', 'memgas', 'memgas_turn', 'memgas_session2turn'):
            f1_values = []
            latency_values = []
            for x in results:
                if '_abs' not in str(x['conversation_id']):
                    if 'f1_score' in x:
                        f1_values.append(x['f1_score'])
                    if 'latency_ms' in x:
                        latency_values.append(x['latency_ms'])
            if f1_values:
                print(f"  F1 Score (avg):     {np.mean(f1_values)*100:.2f}%")
            if latency_values:
                total_sec = sum(latency_values) / 1000
                print(f"  Latency (avg):      {np.mean(latency_values):.1f} ms / question")
                print(f"  Latency (total):    {total_sec:.1f} s ({total_sec/60:.1f} min) for {len(latency_values)} questions")
        print("=" * 80 + "\n")
    

    # save results
    os.makedirs("../../retrieval_logs/", exist_ok=True)
    out_file=f"../../retrieval_logs/{args.dataset}-{args.retriever}-{args.method}.jsonl"
    out_f = open(out_file, 'w')
    for entry in results:
        print(json.dumps(entry), file=out_f)
    out_f.close()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
