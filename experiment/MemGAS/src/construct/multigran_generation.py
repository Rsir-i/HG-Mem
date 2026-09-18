import os
import sys
import json
from glob import glob
from tqdm import tqdm

# Add src/ to sys.path so that `local_qwen` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_qwen import get_local_llm

# ── Conservative threshold: sessions longer than this are skipped to avoid
#    KV-cache OOM ──
MAX_SESSION_CHARS = 20000


def summarize_session(entry, llm, instru_prompt):
    """
    Generate a summary / keywords with the local model.
    Returns None when the session is too long and gets skipped.
    """
    if len(entry) > MAX_SESSION_CHARS:
        print(f"  [skip] session too long ({len(entry)} chars), skipping generation")
        return None
    prompt = f"{instru_prompt}\n\n{entry}\n\nYour answer:"
    return llm.generate(prompt, max_tokens=500, temperature=0.0)


def get_instru_prompt(dataset, level):
    """Return the instruction prompt for the given dataset and granularity."""
    if dataset == 'locomo10':
        if level == 'summary_level':
            return (
                "Below is an user-user dialogue memory. "
                "Please summarize the following dialogue as concisely as possible "
                "in a short paragraph, extracting the main themes and key information.\n"
            )
        elif level == 'keyword_level':
            return (
                "Below is an user-user dialogue memory. "
                "Please extract the most relevant keywords, separated by semicolon.\n"
            )
    else:
        if level == 'summary_level':
            return (
                "Below is an user-AI assistant dialogue memory. "
                "Please summarize the following dialogue as concisely as possible "
                "in a short paragraph, extracting the main themes and key information.\n"
            )
        elif level == 'keyword_level':
            return (
                "Below is an user-AI assistant dialogue memory. "
                "Please extract the most relevant keywords, separated by semicolon.\n"
            )
    raise ValueError(f"Unknown level: {level}")


def granularity_generate(dataset, level, model_path=None):
    """
    Generate summaries / keywords conversation by conversation.

    Each conversation is processed and written to disk independently under
    by_conv/; the per-conversation files are merged into the final jsonl at the
    end. Overly long sessions are skipped instead of being generated.
    """
    llm = get_local_llm(model_path)
    in_data = json.load(open(f'../../data/process_data/{dataset}.json'))

    # Normalize dataset name for output
    save_dataset = 'longmemeval' if 'longmemeval' in dataset else dataset

    # Per-conversation output dir: multi_granularity_logs/{ds}-{level}_by_conv/
    conv_dir = f'../../multi_granularity_logs/{save_dataset}-{level}_by_conv'
    os.makedirs(conv_dir, exist_ok=True)

    # Final merged file (keeps the original format, compatible with construct_emb)
    final_path = f'../../multi_granularity_logs/{save_dataset}-{level}.jsonl'

    instru_prompt = get_instru_prompt(dataset, level)

    total_skipped = 0
    total_sessions = 0
    new_generated = 0

    for sample in tqdm(in_data, desc=f"Generating {level}"):
        conv_id = sample["conversation_id"]
        conv_file = os.path.join(conv_dir, f"{conv_id}.jsonl")

        # Skip conversations that are already finished
        if os.path.exists(conv_file):
            continue

        results = []
        for sessid, sess in zip(sample['sessions_ids'], sample['sessions']):
            total_sessions += 1

            if 'longmemeval' in dataset:
                key = sessid
            else:
                key = f'convid-{str(conv_id)}-sessid-{sessid}'

            entry = '\n\n'.join(sess)
            expansion = summarize_session(entry, llm, instru_prompt)

            if expansion is None:
                # Too long: use the first 200 characters of the session as a
                # placeholder so that downstream code does not fail
                total_skipped += 1
                expansion = f"[SKIPPED: session too long ({len(entry)} chars)] " + entry[:200]
            else:
                new_generated += 1

            results.append({key: expansion})

        # Write each conversation to disk independently
        with open(conv_file, 'w', encoding='utf-8') as f:
            f.writelines([json.dumps(r, ensure_ascii=False) + "\n" for r in results])

    if total_skipped:
        print(f"[warning] {total_skipped}/{total_sessions} sessions skipped as too long")
    if new_generated:
        print(f"[generate] {new_generated} sessions newly generated, "
              f"{total_sessions - new_generated - total_skipped} taken from cache")

    # Merge all per-conversation files
    merge_results(conv_dir, final_path)


def merge_results(conv_dir, final_path):
    """Merge all per-conversation jsonl files into one final file."""
    conv_files = sorted(glob(os.path.join(conv_dir, '*.jsonl')))
    if not conv_files:
        print(f"[warning] no per-conversation files in {conv_dir}")
        return

    with open(final_path, 'w', encoding='utf-8') as f_out:
        for fpath in conv_files:
            with open(fpath, 'r', encoding='utf-8') as f_in:
                f_out.write(f_in.read())

    print(f"[merge] {len(conv_files)} conversations -> {final_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='locomo10, longmemeval_s, longmemeval_m')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the local Qwen-7B-Instruct model '
                             '(can also be set through the LOCAL_MODEL_PATH environment variable)')
    args = parser.parse_args()

    if args.model_path:
        os.environ["LOCAL_MODEL_PATH"] = args.model_path

    granularity_generate(args.dataset, 'summary_level')
    granularity_generate(args.dataset, 'keyword_level')
