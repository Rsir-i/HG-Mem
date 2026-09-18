# Datasets

Two long-term conversational-memory benchmarks are used to evaluate HG-Mem.
Their raw JSON files are **not** tracked by git (the LongMemEval split alone is
≈ 265 MB, far above GitHub's 100 MB per-file hard limit). Download them once
into this folder with the links below; both are consumed **verbatim**, with no
preprocessing or conversion step.

## Download

| Benchmark | File | Size | Download link |
| --------- | ---- | ---- | ------------- |
| LoCoMo (10 conversations) | `locomo10.json` | ≈ 2.7 MB | <https://huggingface.co/datasets/KimmoZZZ/locomo/resolve/main/locomo10.json> |
| LongMemEval (500 items) | `longmemeval_s_cleaned.json` | ≈ 265 MB | <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json> |

From the repository root:

```bash
curl -L -o dataset/locomo10.json \
    https://huggingface.co/datasets/KimmoZZZ/locomo/resolve/main/locomo10.json

curl -L -o dataset/longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Notes:

- The official LoCoMo release lives in the
  [snap-research/locomo](https://github.com/snap-research/locomo) repository
  (`data/locomo10.json`); the HuggingFace URL above serves the identical file
  (2,805,274 bytes).
- The LongMemEval release also ships the `longmemeval_m_cleaned` and
  `longmemeval_oracle` splits — just replace the file name in that URL.
- If `huggingface.co` is not reachable, substitute `hf-mirror.com` for the host
  part of either link.

## 1. `locomo10.json` — LoCoMo (10 conversations)

The LoCoMo benchmark evaluates information retrieval, event understanding,
and temporal reasoning across long time spans and multi-session dialogue
history. We use the 10-conversation split distributed by the original authors.

- **Format**: JSON list, one entry per conversation. Each entry contains the
  following fields:
  - `sample_id` (str): unique identifier of the conversation.
  - `conversation` (dict): mapping from `session_1`, `session_2`, … to a list
    of turn dicts `{dia_id, speaker, text}`.
  - `session_summary` (dict): session-level summaries keyed by
    `session_<i>_summary` (these are the official LoCoMo annotations, used
    by HG-Mem on LoCoMo).
  - `event_summary` (dict): event-level annotations from LoCoMo (not used
    by HG-Mem itself; retained for reference).
  - `observation` (dict): additional LoCoMo observations (not used).
  - `qa` (list): question-answer pairs, each with `question`, `answer`,
    `category` (1: single-hop, 2: temporal, 3: inference, 4: multi-hop,
    5: adversarial), and `evidence` (list of `dia_id`s).

- **Gold evidence**: the list of `dia_id`s in the `evidence` field of each QA
  pair — they are directly used as the relevant-turn set.

- **Source / License**: Maharana et al., *Evaluating Very Long-Term
  Conversational Memory of LLM Agents*, ACL 2024. Please consult the original
  LoCoMo release for its license and redistribution terms.

## 2. `longmemeval_s_cleaned.json` — LongMemEval (500 items)

LongMemEval evaluates long-term interactive memory from multiple
perspectives: information extraction, cross-session reasoning, temporal
reasoning, knowledge updating, and user-preference retention. We use the
official `longmemeval_s_cleaned` split of 500 items.

- **Format**: JSON list, one entry per question. Each entry contains:
  - `question_id` (str): unique identifier of the question.
  - `question_type` (str): one of `single-session-user`,
    `single-session-assistant`, `single-session-preference`,
    `knowledge-update`, `temporal-reasoning`, `multi-session`.
  - `question` (str): the query text.
  - `question_date` (str): the date the question was asked.
  - `answer` (str): the reference answer (used for end-to-end F1 / EM).
  - `answer_session_ids` (list): the haystack session IDs that contain the
    answer.
  - `haystack_session_ids` (list): IDs of all haystack sessions for this
    question.
  - `haystack_dates` (list): dates of the haystack sessions.
  - `haystack_sessions` (list of lists of message dicts): the haystack
    conversation turns. Each message has `role` (`user`/`assistant`),
    `content`, and `has_answer` (boolean flag identifying the gold turn).

- **Gold evidence**: every message with `has_answer == true` in the
  corresponding haystack session. Turn IDs are constructed as
  `"<session_idx>_<msg_idx>"`.

- **Source / License**: Wu et al., *LongMemEval: Benchmarking Chat
  Assistants on Long-Term Interactive Memory*, ICLR 2025. The cleaned
  release is published by the original authors on HuggingFace under the
  **MIT License**
  (<https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>); it
  differs from the first LongMemEval release only in that noisy haystack
  sessions interfering with answer correctness were removed.

## Notes

- Both files are used as-is: no preprocessing beyond the format already in use
  by the experiment scripts is required.
- Session summaries for LongMemEval are **not** included in the released
  JSON. HG-Mem generates them on-the-fly with Qwen2.5-7B-Instruct and
  caches the result to `longmemeval_summaries_v3.json` in the working
  directory. The same applies to turn topics
  (`turn_topics_cache_v3.json`) for LongMemEval and
  (`turn_topics_cache.json`) for LoCoMo.
