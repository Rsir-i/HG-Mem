# SeCom — baseline

**On Memory Construction and Retrieval for Personalized Conversational Agents**

- Paper (ICLR 2025): <https://www.arxiv.org/abs/2502.05589>
- Project page: <https://llmlingua.com/secom.html>
- Upstream repository: <https://github.com/microsoft/SeCom>

This folder holds a copy of the official SeCom release, trimmed to the code needed as a
baseline memory system in our HG-Mem comparison.

SeCom segments a conversation into **topical units**, compresses them with
[LLMLingua-2](https://llmlingua.com/llmlingua2.html) and retrieves memory units from the
compressed representation. It reports strong results on long-term conversation benchmarks
such as LoCoMo and Long-MT-Bench+.

## Contents

| Path | Description |
| --- | --- |
| `secom/` | The installable package (`SeCom` API, version, retrieval configs, segmentation prompts) |
| `experiment/run_eval.py` | **Main evaluation pipeline used for the paper** (LoCoMo / LongMemEval, Qwen-7B) |
| `experiment/segment.py` | Topic segmentation of conversations |
| `experiment/compress.py` | Prompt compression of the segmented units |
| `experiment/retrieve.py` | Retrieval over the compressed memory units |
| `experiment/chat.py` | Answer generation for the retrieval outputs |
| `experiment/metrics.py`, `experiment/utils.py` | Metric computation and shared helpers |
| `experiment/download_data.py` | Downloads Long-MT-Bench+ (not required for LoCoMo/LongMemEval) |
| `experiment/run.sh` | Reference end-to-end pipeline script |
| `setup.py` | Package definition (`pip install -e .`) |
| `requirements.txt` | Dependencies |

> Datasets are **not** duplicated in this folder: both benchmarks are downloaded once
> into the repository-level `dataset/` folder (see
> [`../../dataset/README.md`](../../dataset/README.md)) and are selected with
> `--data_dir`.

## Install

```bash
pip install llmlingua
pip install -e .
pip install python-dotenv
```

SeCom reads the LLM credentials from a dotenv file. Put your key and base URL in
`~/dot_env/openai.env`:

```
OPENAI_API_KEY=""
OPENAI_API_BASE=""
```

## Running the evaluation

`run_eval.py` requires a local model directory through `--model_path`. It expects the
dataset files in `--data_dir`; passing the repository-level `dataset/` folder makes both
benchmarks available:

LoCoMo:

```bash
cd experiment
python run_eval.py \
    --model_path /models/Qwen2.5-7B-Instruct \
    --dataset locomo10 \
    --data_dir ../../dataset
```

LongMemEval:

```bash
cd experiment
python run_eval.py \
    --model_path /models/Qwen2.5-7B-Instruct \
    --dataset longmemeval \
    --data_dir ../../dataset
```

`--data_dir` must be given explicitly if you run from a different working directory;
otherwise it defaults to a folder that is not shipped here. Add `--no_segmentation` to
fall back to session-level memory units, or `--embedding_model` to change the retrieval
embedder (default `sentence-transformers/all-MiniLM-L6-v2`).

For the Long-MT-Bench+ pipeline used upstream, see `experiment/run.sh`.

## Citation

```bibtex
@inproceedings{secom2025,
  title     = {On Memory Construction and Retrieval for Personalized Conversational Agents},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

Please refer to the paper (<https://www.arxiv.org/abs/2502.05589>) for the complete
author list.

## License

This folder is a copy of the upstream Microsoft SeCom release and is distributed under
its original **MIT License** — see [`LICENSE`](LICENSE).
