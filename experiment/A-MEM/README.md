# A-Mem — baseline (Agentic Memory)

This folder holds a copy of the official A-Mem reproduction release, trimmed to the code
needed as a baseline memory system in our HG-Mem comparison.

- Paper: *A-Mem: Agentic Memory for LLM Agents* — <https://arxiv.org/pdf/2502.12110>
- Upstream code: <https://github.com/WujiangXu/AgenticMemory>
- Official product implementation: <https://github.com/WujiangXu/A-mem-sys>

The code is kept **only** so that the baseline numbers reported in our paper can be
reproduced. For building your own agents on top of A-Mem, please use the upstream
[A-mem-sys](https://github.com/WujiangXu/A-mem-sys) repository instead.

## Contents

| Path | Description |
| --- | --- |
| `memory_layer.py` | Core memory layer: note construction, linking, retrieval helpers |
| `memory_layer_robust.py` | Robust variant used by the evaluation scripts (`memory_layer` is imported by it) |
| `llm_text_parsers.py` | Parsers that turn LLM text into structured memory notes |
| `load_dataset.py` | Dataset loaders for LoCoMo and LongMemEval |
| `utils.py` | Shared helpers (LLM clients, embedding clients, logging) |
| `test_advanced.py` | Original LoCoMo evaluation (requires an OpenAI-style JSON-schema backend) |
| `test_advanced_robust.py` | **LoCoMo evaluation used for the paper** (works with local vLLM/SGLang backends) |
| `test_longmemeval_robust.py` | **LongMemEval evaluation used for the paper** |
| `totaltime_longmemeval.py` | Post-processing helper that totals model runtime from an evaluation log |
| `run_all_experiments.sh`, `run_k_sweep.sh` | Reference driver scripts (they contain cluster-specific paths — edit before use) |
| `requirements.txt`, `requirements_longmemeval.txt` | Dependencies |

> Datasets are **not** duplicated in this folder: both benchmarks are downloaded once
> into the repository-level `dataset/` folder (see
> [`../../dataset/README.md`](../../dataset/README.md)).

## Environment

```bash
pip install -r requirements.txt
pip install -r requirements_longmemeval.txt   # extras needed by the LongMemEval script
```

`--backend` selects the LLM provider: `openai`, `ollama`, `sglang`, `vllm` or `transformers`.
For a fully local run, serve an OpenAI-compatible model (e.g. with vLLM) and point
`--sglang_host` / `--sglang_port` at it.

## Running the evaluation

`--dataset` and `--output` are resolved **relative to this folder** (the script joins them
with its own directory), so `../../dataset/...` refers to the repository-level `dataset/`
folder.

LoCoMo:

```bash
python test_advanced_robust.py \
    --dataset ../../dataset/locomo10.json \
    --backend vllm --model Qwen/Qwen2.5-7B-Instruct --sglang_port 30000 \
    --retrieve_k 10 --output amem_locomo.json
```

LongMemEval:

```bash
python test_longmemeval_robust.py \
    --dataset ../../dataset/longmemeval_s_cleaned.json \
    --backend vllm --model Qwen/Qwen2.5-7B-Instruct --sglang_port 30000 \
    --retrieve_k 10 --embed_model all-MiniLM-L6-v2 \
    --output amem_longmemeval.json
```

Use `--ratio 0.1` for a quick small-sample validation run before a full evaluation.

## Citation

```bibtex
@article{xu2025amem,
  title   = {A-Mem: Agentic Memory for LLM Agents},
  author  = {Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal = {arXiv preprint arXiv:2502.12110},
  year    = {2025}
}
```

## License

This code is a copy of the upstream release and is distributed under its original
**MIT License** — see [`LICENSE`](LICENSE).
