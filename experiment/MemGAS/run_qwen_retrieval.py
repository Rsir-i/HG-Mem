"""
MemGAS + local Qwen-7B-Instruct retrieval pipeline
==================================================
Loads a local Qwen-7B-Instruct model (transformers) to replace GPT-4o/GPT-4o-mini
and evaluates retrieval metrics only on locomo10 and longmemeval_s:
R@1, R@3, R@5, R@10, MRR. Small-sample experiments are supported (--few_shot N).

Usage:
  # Full experiment
  python run_qwen_retrieval.py --model_path /path/to/Qwen2.5-7B-Instruct

  # Explicit dataset paths
  python run_qwen_retrieval.py --model_path /path/to/Qwen2.5-7B-Instruct \\
      --locomo10_path /data/locomo10.json \\
      --longmemeval_s_path /data/longmemeval_s_cleaned.json

  # locomo10 only
  python run_qwen_retrieval.py --model_path /path/to/Qwen2.5-7B-Instruct \\
      --locomo10_path /data/locomo10.json --experiments locomo10

  # Small sample: use only the first 5 conversations of each dataset
  python run_qwen_retrieval.py --model_path /path/to/Qwen2.5-7B-Instruct \\
      --locomo10_path /data/locomo10.json --few_shot 5

  # Select retrieval methods
  python run_qwen_retrieval.py --model_path /path/to/Qwen2.5-7B-Instruct \\
      --locomo10_path /data/locomo10.json --methods memgas session_level
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import subprocess
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ORIGIN_DATA_DIR = DATA_DIR / "origin_data"
PROCESS_DATA_DIR = DATA_DIR / "process_data"
MULTI_GRAN_DIR = BASE_DIR / "multi_granularity_logs"
EMB_DIR = DATA_DIR / "process_embs"
GRAPH_DIR = BASE_DIR / "graph_cache"
RETRIEVAL_DIR = BASE_DIR / "retrieval_logs"

DEFAULT_METHODS = ["session_level", "turn_level", "turn_retrieval",
                   "summary_level", "keyword_level", "hybrid_level",
                   "lite", "memgas", "memgas_turn"]
DEFAULT_K_VALUES = [1, 3, 5, 10]
ALL_DATASETS = ["locomo10", "longmemeval_s"]

# MemGAS hyper-parameters
MEM_THRESHOLD = 30
N_COMPONENTS = 2
NUM_SEEDNODES = 15
DAMPING = 0.1
TEMP = 0.1


# ─── Datasets — name → default path ──────────────────────────────────────

def _default_locomo10_path():
    p = BASE_DIR / "long-term-memory" / "locomo10.json"
    return str(p) if p.exists() else None


def _default_longmemeval_path():
    p = BASE_DIR / "long-term-memory" / "longmemeval_s_cleaned.json"
    return str(p) if p.exists() else None


def ensure_dirs(dataset_paths: dict):
    """Create the required directories and copy the raw data into origin_data/."""
    ORIGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_GRAN_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    # locomo10 → data/origin_data/locomo10.json
    if "locomo10" in dataset_paths and dataset_paths["locomo10"]:
        src = Path(dataset_paths["locomo10"])
        dst = ORIGIN_DATA_DIR / "locomo10.json"
        if not src.exists():
            raise FileNotFoundError(f"locomo10 data file not found: {src}")
        if not dst.exists():
            print(f"[data] copying {src} → {dst}")
            shutil.copy2(str(src), str(dst))

    # longmemeval_s → data/origin_data/longmemeval_s (no extension, as in the original code)
    if "longmemeval_s" in dataset_paths and dataset_paths["longmemeval_s"]:
        src = Path(dataset_paths["longmemeval_s"])
        dst = ORIGIN_DATA_DIR / "longmemeval_s"
        if not src.exists():
            raise FileNotFoundError(f"longmemeval_s data file not found: {src}")
        if not dst.exists():
            print(f"[data] copying {src} → {dst}")
            shutil.copy2(str(src), str(dst))


def run_preprocess(datasets):
    """Run data preprocessing."""
    print("\n" + "=" * 80)
    print("[stage 1] Data preprocessing")
    print("=" * 80)

    from data.dataprocess import process_locomo10, process_longmemeval

    if "locomo10" in datasets:
        output = PROCESS_DATA_DIR / "locomo10.json"
        if output.exists():
            print(f"[skip] {output} already exists")
        else:
            print("[run] process_locomo10")
            process_locomo10()

    if "longmemeval_s" in datasets:
        output = PROCESS_DATA_DIR / "longmemeval_s.json"
        if output.exists():
            print(f"[skip] {output} already exists")
        else:
            print("[run] process_longmemeval (s)")
            process_longmemeval()


def apply_few_shot(dataset, n_samples):
    """Small sample: keep only the first N conversations."""
    data_path = PROCESS_DATA_DIR / f"{dataset}.json"
    if not data_path.exists():
        print(f"[warning] {data_path} does not exist, cannot take a small sample")
        return None

    few_shot_path = PROCESS_DATA_DIR / f"{dataset}_fewshot{n_samples}.json"
    if few_shot_path.exists():
        print(f"[few-shot] {few_shot_path} already exists, skipping")
        return few_shot_path

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subset = data[:n_samples]
    with open(few_shot_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print(f"[few-shot] {dataset}: {len(data)} → {len(subset)} conversations, saved to {few_shot_path}")
    return few_shot_path


def run_generation(dataset, skip_if_exists=True):
    """
    Generate summaries and keywords with the local model.
    Runs multigran_generation.py through a subprocess (it works from the
    src/construct/ directory).
    """
    print(f"\n[stage 2] Multi-granularity generation: {dataset}")

    save_dataset = "longmemeval" if "longmemeval" in dataset else dataset
    summary_path = MULTI_GRAN_DIR / f"{save_dataset}-summary_level.jsonl"
    keyword_path = MULTI_GRAN_DIR / f"{save_dataset}-keyword_level.jsonl"

    if skip_if_exists and summary_path.exists() and keyword_path.exists():
        print(f"[skip] summary and keyword for {save_dataset} already exist")
        return

    gen_script = BASE_DIR / "src" / "construct" / "multigran_generation.py"
    env = os.environ.copy()
    # LOCAL_MODEL_PATH is set in main()

    cmd = [sys.executable, str(gen_script), "--dataset", dataset]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(gen_script.parent), env=env, check=True)


def run_embedding(dataset, retriever):
    """Build embeddings (invokes emb_rawdata through a subprocess)."""
    print(f"\n[stage 3] Embedding construction: {dataset} ({retriever})")

    emb_path = EMB_DIR / f"{dataset}-{retriever}-emb.pt"
    if emb_path.exists():
        print(f"[skip] {emb_path} already exists")
        return

    cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0,'.'); from construct_emb import emb_rawdata; emb_rawdata('{dataset}','{retriever}')"
    ]
    cwd = str(BASE_DIR / "src" / "construct")
    print(f"[run] emb_rawdata('{dataset}', '{retriever}')")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_graph_construction(dataset, retriever):
    """Build the association graph."""
    print(f"\n[stage 4] Association graph construction: {dataset}")

    graph_path = GRAPH_DIR / f"graph-{dataset}-{retriever}-{MEM_THRESHOLD}-{N_COMPONENTS}.pt"
    if graph_path.exists():
        print(f"[skip] {graph_path} already exists")
        return

    asso_script = BASE_DIR / "src" / "construct" / "construct_asso.py"
    cmd = [
        sys.executable, str(asso_script),
        "--dataset", dataset,
        "--retriever", retriever,
        "--mem_threshold", str(MEM_THRESHOLD),
        "--n_components", str(N_COMPONENTS),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(asso_script.parent), check=True)


def run_graph_construction_turn(dataset, retriever):
    """Build the turn-level association graph (memgas_turn)."""
    print(f"\n[stage 4b] Turn-level association graph construction: {dataset}")

    graph_path = GRAPH_DIR / f"graph-turn-{dataset}-{retriever}-{MEM_THRESHOLD}-{N_COMPONENTS}.pt"
    if graph_path.exists():
        print(f"[skip] {graph_path} already exists")
        return

    asso_script = BASE_DIR / "src" / "construct" / "construct_asso.py"
    cmd = [
        sys.executable, str(asso_script),
        "--dataset", dataset,
        "--retriever", retriever,
        "--mode", "turn",
        "--mem_threshold", str(MEM_THRESHOLD),
        "--n_components", str(N_COMPONENTS),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(asso_script.parent), check=True)


def run_retrieval(dataset, methods, retriever):
    """Run the retrieval evaluation."""
    print(f"\n[stage 5] Retrieval evaluation: {dataset}")

    ret_script = BASE_DIR / "src" / "retrieval" / "run_retrieval.py"

    for method in methods:
        print(f"\n--- method: {method} ---")
        cmd = [
            sys.executable, str(ret_script),
            "--dataset", dataset,
            "--retriever", retriever,
            "--method", method,
            "--num_seednodes", str(NUM_SEEDNODES),
            "--mem_threshold", str(MEM_THRESHOLD),
            "--n_components", str(N_COMPONENTS),
            "--damping", str(DAMPING),
            "--temp", str(TEMP),
        ]
        subprocess.run(cmd, cwd=str(ret_script.parent), check=True)


def print_summary(datasets, methods, k_values, retriever):
    """Aggregate and print the retrieval metrics."""
    print("\n" + "=" * 100)
    print(f"                       Final retrieval metrics summary ({retriever})")
    print("=" * 100)

    header = f"{'Dataset':<18} {'Method':<20}"
    for k in k_values:
        header += f"  {'R@'+str(k):>8}"
    header += f"  {'MRR':>8}  {'F1':>8}  {'Latency':>10}"
    print(header)
    print("-" * 110)

    for dataset in datasets:
        for method in methods:
            log_path = RETRIEVAL_DIR / f"{dataset}-{retriever}-{method}.jsonl"
            if not log_path.exists():
                print(f"{dataset:<18} {method:<20} [no retrieval results found]")
                continue

            results = []
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line.strip()))

            if not results:
                continue

            row = f"{dataset:<18} {method:<20}"
            # turn_retrieval / memgas_turn use metrics['turn'], others use metrics['session']
            metric_src = 'turn' if method in ('turn_retrieval', 'memgas_turn', 'memgas_session2turn') else 'session'
            metrics_dict = results[0].get("retrieval_results", {}).get("metrics", {}).get(metric_src, {})
            for k in k_values:
                r_key = f"recall@{k}"
                if r_key in metrics_dict:
                    val = 0
                    count = 0
                    for x in results:
                        if "_abs" not in str(x.get("conversation_id", "")):
                            m = x["retrieval_results"]["metrics"][metric_src]
                            if r_key in m:
                                val += m[r_key]
                                count += 1
                    val = val / count if count > 0 else 0
                    row += f"  {val*100:6.2f}%"
                else:
                    row += f"  {'N/A':>8}"

            mrr_k = f"mrr@{max(k_values)}"
            if mrr_k in metrics_dict:
                val = 0
                count = 0
                for x in results:
                    if "_abs" not in str(x.get("conversation_id", "")):
                        m = x["retrieval_results"]["metrics"][metric_src]
                        if mrr_k in m:
                            val += m[mrr_k]
                            count += 1
                val = val / count if count > 0 else 0
                row += f"  {val*100:6.2f}%"
            else:
                row += f"  {'N/A':>8}"

            # ── F1 and average latency (turn_retrieval / memgas / memgas_turn / memgas_session2turn) ──
            if method in ('turn_retrieval', 'memgas', 'memgas_turn', 'memgas_session2turn'):
                f1_vals = [x.get('f1_score', None) for x in results
                           if "_abs" not in str(x.get("conversation_id", "")) and 'f1_score' in x]
                lat_vals = [x.get('latency_ms', None) for x in results
                            if "_abs" not in str(x.get("conversation_id", "")) and 'latency_ms' in x]
                avg_f1 = np.mean(f1_vals) if f1_vals else 0
                avg_lat = np.mean(lat_vals) if lat_vals else 0
                row += f"  {avg_f1*100:6.2f}%  {avg_lat:8.1f}ms"
            else:
                row += f"  {'N/A':>8}  {'N/A':>10}"

            print(row)

    print("-" * 110)
    print("\nDetailed metrics (including R@30, R@50):")
    for dataset in datasets:
        for method in methods:
            log_path2 = RETRIEVAL_DIR / f"{dataset}-{retriever}-{method}.jsonl"
            if not log_path2.exists():
                continue
            results = []
            with open(log_path2, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            if not results:
                continue
            print(f"\n[{dataset}] {method}:")
            metric_src2 = 'turn' if method in ('turn_retrieval', 'memgas_turn', 'memgas_session2turn') else 'session'
            metric_src_data = results[0]['retrieval_results']['metrics'][metric_src2]
            if not metric_src_data:
                print("  (no metrics)")
                continue
            metric_names = metric_src_data.keys()
            for k_name in sorted(metric_names):
                k_result = 0
                count = 0
                for x in results:
                    if "_abs" not in str(x.get("conversation_id", "")):
                        m = x["retrieval_results"]["metrics"][metric_src2]
                        if k_name in m:
                            k_result += m[k_name]
                            count += 1
                k_result = k_result / count if count > 0 else 0
                print(f"  {k_name}: {k_result*100:.2f}%")
            # ── End-to-end F1 + latency (turn_retrieval / memgas / memgas_turn) ──
            if method in ('turn_retrieval', 'memgas', 'memgas_turn'):
                f1_vals = [x.get('f1_score', None) for x in results
                           if "_abs" not in str(x.get("conversation_id", "")) and 'f1_score' in x]
                lat_vals = [x.get('latency_ms', None) for x in results
                            if "_abs" not in str(x.get("conversation_id", "")) and 'latency_ms' in x]
                if f1_vals:
                    print(f"  F1 Score (avg): {np.mean(f1_vals)*100:.2f}%")
                if lat_vals:
                    print(f"  Latency (avg):  {np.mean(lat_vals):.1f} ms / question")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MemGAS + local Qwen-7B-Instruct retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use the default data path (long-term-memory/)
  python run_qwen_retrieval.py --model_path /models/Qwen2.5-7B-Instruct

  # Explicit dataset file paths
  python run_qwen_retrieval.py --model_path /models/Qwen2.5-7B-Instruct \\
      --locomo10_path /data/locomo10.json \\
      --longmemeval_s_path /data/longmemeval_s_cleaned.json

  # locomo10 only
  python run_qwen_retrieval.py --model_path /models/Qwen2.5-7B-Instruct \\
      --locomo10_path ./locomo10.json --experiments locomo10

  # Small sample
  python run_qwen_retrieval.py --model_path /models/Qwen2.5-7B-Instruct \\
      --locomo10_path ./locomo10.json --few_shot 5
        """,
    )
    # ── Model arguments ──
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the local Qwen-7B-Instruct model (e.g. /models/Qwen2.5-7B-Instruct)")

    # ── Dataset paths ──
    parser.add_argument("--locomo10_path", type=str, default=None,
                        help="Path to locomo10.json (falls back to long-term-memory/locomo10.json if not given)")
    parser.add_argument("--longmemeval_s_path", type=str, default=None,
                        help="Path to longmemeval_s_cleaned.json (falls back to long-term-memory/longmemeval_s_cleaned.json if not given)")

    # ── Experiment scope ──
    parser.add_argument("--retriever", type=str, default="contriever",
                        help="Embedding model: contriever (default), mpnet, minilm, qaminilm")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help=f"Datasets to run, default {ALL_DATASETS} (options: locomo10, longmemeval_s)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help=f"Retrieval methods, default {DEFAULT_METHODS}")
    parser.add_argument("--eval_k", type=int, nargs="+", default=None,
                        help=f"Evaluation K values, default {DEFAULT_K_VALUES}")

    parser.add_argument("--few_shot", type=int, default=None,
                        help="Small-sample size (use only the first N conversations of each dataset)")

    # ── Stage control ──
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip_generation", action="store_true", help="Skip summary/keyword generation")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding construction")
    parser.add_argument("--skip_graph", action="store_true", help="Skip association graph construction")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval (only summarize existing results)")

    args = parser.parse_args()

    # ── Validate the model path ──
    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        print(f"[error] model path does not exist: {model_path}")
        sys.exit(1)

    # ── Set environment variables (all sub-modules load the model via LOCAL_MODEL_PATH) ──
    os.environ["LOCAL_MODEL_PATH"] = model_path
    # Add src/ to PYTHONPATH so that sub-modules can import local_qwen
    src_dir = str(BASE_DIR / "src")
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = src_dir + os.pathsep + os.environ["PYTHONPATH"]
    else:
        os.environ["PYTHONPATH"] = src_dir
    sys.path.insert(0, src_dir)

    # ── Determine dataset paths and experiment scope ──
    # Logic:
    #   - --experiments given → run only those; prefer explicit paths, fall back to defaults
    #   - --experiments not given → run only the datasets for which a --xxx_path was given
    #   - neither given → look up the default paths and run every dataset that was found

    dataset_paths = {}

    # Collect explicitly given paths
    if args.locomo10_path:
        dataset_paths["locomo10"] = args.locomo10_path
    if args.longmemeval_s_path:
        dataset_paths["longmemeval_s"] = args.longmemeval_s_path

    if args.experiments:
        # Experiment scope was given; fill in default paths for datasets without one
        experiments = args.experiments
        for ds in experiments:
            if ds not in dataset_paths:
                if ds == "locomo10":
                    p = _default_locomo10_path()
                elif ds == "longmemeval_s":
                    p = _default_longmemeval_path()
                else:
                    p = None
                if p:
                    dataset_paths[ds] = p
                else:
                    print(f"[warning] no data file found for {ds}, it will be skipped")
    elif dataset_paths:
        # Only paths were given, no --experiments → run only those with a path
        experiments = list(dataset_paths.keys())
    else:
        # Nothing given → search the default paths
        p = _default_locomo10_path()
        if p:
            dataset_paths["locomo10"] = p
        else:
            print("[warning] locomo10.json not found, the locomo10 experiment will be skipped")
        p = _default_longmemeval_path()
        if p:
            dataset_paths["longmemeval_s"] = p
        else:
            print("[warning] longmemeval_s_cleaned.json not found, the longmemeval_s experiment will be skipped")
        experiments = list(dataset_paths.keys())

    # Filter: keep only datasets with a data file
    experiments = [ds for ds in experiments if ds in dataset_paths]
    if not experiments:
        print("[error] no usable dataset; specify one via --locomo10_path / --longmemeval_s_path")
        sys.exit(1)

    methods = args.methods or DEFAULT_METHODS
    k_values = args.eval_k or DEFAULT_K_VALUES
    retriever = args.retriever

    # ── Print the configuration ──
    print("=" * 80)
    print("  MemGAS + local Qwen-7B-Instruct retrieval experiment")
    print("=" * 80)
    print(f"  Model path:        {model_path}")
    print(f"  Datasets:          {experiments}")
    for ds in experiments:
        print(f"    {ds}: {dataset_paths[ds]}")
    print(f"  Retrieval methods: {methods}")
    print(f"  Evaluation K:      {k_values}")
    print(f"  Embedding:         {retriever}")
    if args.few_shot:
        print(f"  Few-shot:          N={args.few_shot}")
    print("=" * 80)

    # 1. Data preparation
    ensure_dirs(dataset_paths)

    # 2. Data preprocessing
    if not args.skip_preprocess:
        run_preprocess(experiments)

    # 3. Small-sample selection
    active_datasets = list(experiments)
    if args.few_shot:
        new_datasets = []
        for ds in experiments:
            few_path = apply_few_shot(ds, args.few_shot)
            if few_path:
                new_datasets.append(f"{ds}_fewshot{args.few_shot}")
        if new_datasets:
            active_datasets = new_datasets
            print(f"[few-shot] experiment datasets switched to: {active_datasets}")

    # 4. Summary / keyword generation (with the local model)
    if not args.skip_generation:
        for ds in active_datasets:
            run_generation(ds)

    # 5. Embedding construction
    if not args.skip_embeddings:
        for ds in active_datasets:
            run_embedding(ds, retriever)

    # 6. Association graph construction
    if not args.skip_graph:
        for ds in active_datasets:
            if any(m in methods for m in ("memgas", "memgas_session2turn")):
                run_graph_construction(ds, retriever)
            if "memgas_turn" in methods:
                run_graph_construction_turn(ds, retriever)

    # 7. Retrieval evaluation
    if not args.skip_retrieval:
        for ds in active_datasets:
            run_retrieval(ds, methods, retriever)

    # 8. Summarize the results
    print_summary(active_datasets, methods, k_values, retriever)

    print("Experiment finished!")


if __name__ == "__main__":
    main()
