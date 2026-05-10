# nlp-quantization

*Impact of Quantization Methods on Large Language Model Performance Across NLP Tasks*

A reproducible experiment pipeline that compares three open-source LLMs (**Mistral-7B-v0.3**, **Phi-3-mini-4k**, **LLaMA-3.1-8B**) under four quantization strategies (`fp16`, `bnb_nf4`, `gptq-4bit`, `awq-4bit`) across three NLP tasks (**ARC-Challenge**, **WikiText-2**, **MMLU**).

Each `(model × quantization)` combination runs as a single CLI invocation and appends one row to a single `results.csv`. Full matrix: `3 models × 4 quantizations = 12 rows`.

> Architectural decisions: [`ARCHITECTURE.md`](ARCHITECTURE.md) — Phase-by-phase build guide: [`implementation.md`](implementation.md)

---

## Architecture (ASCII)

```
                              ┌─────────────────────────┐
                              │   configs/*.yaml (12)   │
                              │  model × quantization   │
                              └────────────┬────────────┘
                                           │ --config
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          src/nlp_quantization/cli.py                       │
│                              (orchestration)                               │
│                                                                            │
│   1. load_config ──► ExperimentConfig (pydantic validation)                │
│   2. generate_run_id  ──►  {name}__{quant}__{bits}bit__{YYYYMMDD-HHMM}     │
│   3. set_seeds(torch / numpy / random / transformers)                      │
│   4. track_load()  ── peak VRAM + wall-clock                               │
│   5. LOADER_REGISTRY[quant_type](cfg)                                      │
│   6. evaluate_model(model, tokenizer, run_id)                              │
│   7. append_csv(results/results.csv, row)                                  │
│   8. cleanup_gpu (del model + empty_cache)                                 │
└──────────┬───────────────────────────────────────────────────┬─────────────┘
           │                                                   │
           ▼                                                   ▼
┌────────────────────────┐                     ┌──────────────────────────────┐
│  loaders/  (registry)  │                     │      evaluation/             │
│                        │                     │                              │
│  ┌──────────────────┐  │                     │  ┌────────────────────────┐  │
│  │ fp16.py          │  │                     │  │ tasks.py               │  │
│  │ bnb_nf4.py       │  │   model + tok       │  │  TASKS = [             │  │
│  │ gptq.py          │──┼─────────────────────┼─►│   arc_challenge (25fs) │  │
│  │ awq.py           │  │                     │  │   wikitext       (0fs) │  │
│  └──────────────────┘  │                     │  │   mmlu          (5fs)  │  │
│                        │                     │  │  ]                     │  │
│  LOADER_REGISTRY:      │                     │  └────────┬───────────────┘  │
│   "fp16"    → fp16.load│                     │           │                  │
│   "bnb_nf4" → bnb.load │                     │           ▼                  │
│   "gptq"    → gptq.load│                     │  ┌────────────────────────┐  │
│   "awq"     → awq.load │                     │  │ evaluator.py           │  │
└────────────────────────┘                     │  │  HFLM(model,tokenizer) │  │
                                               │  │  ── reset peak VRAM    │  │
                                               │  │  ── simple_evaluate    │  │
                                               │  │  ── try/except per task│  │
                                               │  └────────┬───────────────┘  │
                                               └───────────┼──────────────────┘
                                                           │
              flat dict: {arc_acc_norm, wikitext_ppl, mmlu_acc, *_peak_vram, *_runtime}
                                                           │
                                                           ▼
                                  ┌──────────────────────────────────────┐
                                  │      results_writer.append_csv       │
                                  │   header bootstrap + flock(LOCK_EX)  │
                                  └──────────────────┬───────────────────┘
                                                     ▼
                                  ┌──────────────────────────────────────┐
                                  │     results/results.csv (1 row)      │
                                  │   run_id, model, quant, metrics, env │
                                  └──────────────────────────────────────┘
```

**Design principle:** registry pattern. Adding a new quantization method = `loaders/<new>.py` + one line in `LOADER_REGISTRY`. Adding a new task = one `TaskSpec` line in `TASKS`. The rest of the pipeline stays untouched.

---

## Repository Layout

```
nlp-quantization/
├── ARCHITECTURE.md              # architectural spec
├── implementation.md            # 3-phase build guide
├── README.md                    # this file
├── pyproject.toml
├── configs/                     # 12 YAMLs, one per experiment
├── src/nlp_quantization/
│   ├── cli.py                   # __main__ orchestration
│   ├── config.py                # pydantic ExperimentConfig
│   ├── run_id.py                # automatic run_id generation
│   ├── metrics.py               # VRAM + timing context managers
│   ├── results_writer.py        # append-only CSV writer (flock)
│   ├── loaders/                 # 4 quantization adapters + registry
│   └── evaluation/
│       ├── tasks.py             # TASKS constant list
│       └── evaluator.py         # evaluate_model(...)
├── scripts/run_matrix.sh        # runs the full matrix sequentially
└── results/results.csv          # gitignored, append-only
```

---

## Installation

### 1. System requirements

- **Python** 3.10+
- **CUDA-capable GPU** (required for real evaluation; dry-run and unit tests work on CPU). At least **~16 GB VRAM** recommended for LLaMA-3.1-8B fp16; **~6–8 GB** suffices for 4-bit quantizations.
- **Linux** is recommended. The `bitsandbytes`, `auto-gptq`, and `autoawq` packages are gated by `sys_platform == 'linux'` in `pyproject.toml` — on Windows/macOS those three loaders are unavailable, but the fp16 baseline still works.

### 2. Virtual environment + install

```bash
# clone the repo
git clone <repo-url> nlp-quantization
cd nlp-quantization

# venv
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# editable install (runtime + dev dependencies)
pip install -e ".[dev]"
```

### 3. (LLaMA only) HuggingFace auth

LLaMA-3.1 is a gated repo. You need access approval on the HF side and a logged-in CLI before downloading:

```bash
huggingface-cli login
```

---

## Usage — Step by Step

### Step 1 — Pick a config (or write a new one)

`configs/` ships with 12 ready experiment files. Each file defines **a single experiment**:

```yaml
# configs/phi3_bnb_nf4.yaml
model:
  path: microsoft/Phi-3-mini-4k-instruct
  display_name: phi-3-mini-4k
  trust_remote_code: false
quantization:
  type: bnb_nf4          # fp16 | bnb_nf4 | gptq | awq
  bit_width: 4           # 16 (fp16), 4 (bnb_nf4/awq), 4|8 (gptq)
```

The `runtime` and `output` blocks are optional; if omitted, defaults are used (`device=cuda`, `batch_size=auto`, `seed=1234`, `csv_path=results/results.csv`).

### Step 2 — Validate the config with a dry-run

Validates the config and emits a `run_id`, without downloading the model or touching the GPU:

```bash
python -m nlp_quantization --config configs/phi3_bnb_nf4.yaml --dry-run
# stdout: phi-3-mini-4k__bnb_nf4__4bit__20260510-1432
```

Exit code `0` means the config is valid and the pipeline is ready to run. If you get a `pydantic.ValidationError`, fix the `bit_width` / `quantization.type` combination per `ARCHITECTURE.md §3`.

### Step 3 — Run a single experiment

```bash
python -m nlp_quantization --config configs/phi3_bnb_nf4.yaml
```

The pipeline:

1. Loads the model via the quantization adapter (peak VRAM and load time are measured).
2. Evaluates ARC-Challenge (25-shot) → WikiText-2 (0-shot) → MMLU (5-shot) sequentially via `lm-evaluation-harness`.
3. Records the **per-task peak VRAM** and **wall-clock runtime** for each task.
4. If a task crashes, that task's three columns become `None` while the others and the CSV append continue.
5. Appends a single row to `results/results.csv`.

> Time estimate: a single experiment takes **~30–90 min** on a GPU including MMLU. The full matrix takes **a few hours to a full day**.

### Step 4 — Override the CSV path (optional)

```bash
python -m nlp_quantization \
  --config configs/phi3_bnb_nf4.yaml \
  --csv-path results/phi3_only.csv
```

### Step 5 — Run the full matrix (12 experiments)

```bash
bash scripts/run_matrix.sh
```

The helper script iterates over `configs/*.yaml` sequentially. If an experiment crashes, it prints `FAILED: <cfg>` and continues with the rest. Each successful invocation appends one row to `results/results.csv`.

### Step 6 — Analyze the results

`results/results.csv` can be loaded straight into pandas:

```python
import pandas as pd
df = pd.read_csv("results/results.csv")

# RQ1: ARC accuracy × quantization
df.groupby("quantization_type")["arc_challenge_acc_norm"].mean()

# RQ2: WikiText perplexity × peak VRAM (Pareto)
df[["model_name", "quantization_type",
    "wikitext_word_perplexity", "wikitext_peak_vram_gb"]]

# RQ3: percent change vs fp16 baseline
baseline = df[df.quantization_type == "fp16"].set_index("model_name")
# ... divide each metric by baseline
```

Full column list: `ARCHITECTURE.md §6`.

---

## CLI Reference

```
python -m nlp_quantization [-h] --config CONFIG [--dry-run] [--csv-path CSV_PATH]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | **Required.** YAML experiment file. |
| `--dry-run` | Validates the config + prints the `run_id`; does not load the model or write CSV. Exits 0. |
| `--csv-path <path>` | Overrides `cfg.output.csv_path`. |

Exit codes: `0` success, `1` pipeline failure (loader / orchestration).

---

## Adding a New Experiment

### New quantization method (e.g. HQQ)

1. Write `src/nlp_quantization/loaders/hqq.py` — `def load(cfg) -> tuple[model, tokenizer]`.
2. Add `LOADER_REGISTRY["hqq"] = hqq.load` in `loaders/__init__.py`.
3. Add the `("hqq", <bit_width>)` combination to the validator in `config.py`.
4. Add `configs/<model>_hqq_<bits>bit.yaml` files.

### New task (e.g. HellaSwag)

1. Add `TaskSpec("hellaswag", 10, "acc_norm")` to the `TASKS` list in `evaluation/tasks.py`.
2. Add three new columns (`hellaswag_acc_norm`, `hellaswag_peak_vram_gb`, `hellaswag_runtime_sec`) to `_build_row` in `cli.py`.
3. The CSV header bootstrap will automatically pick up the new columns.

Details: `ARCHITECTURE.md §9 (Extension Points)`.

---

## Tests

```bash
pytest tests/ -v
```

The test strategy is detailed in `implementation.md`:

- **Phase 1**: pure-Python unit tests (config, run_id, results_writer, tasks).
- **Phase 2**: loader registry + tiny model (`hf-internal-testing/tiny-random-LlamaForCausalLM`) smoke test.
- **Phase 3**: `monkeypatch`-ed `simple_evaluate` for evaluator + error policy + dry-run subprocess test.

Real 7B+ models are **never** downloaded in any test.

---

## Reproducibility

Each CSV row carries the following metadata:

- `seed` (default `1234`, applied via `set_seeds()` to torch/numpy/random/transformers)
- `transformers_version`, `torch_version`, `lm_eval_version`
- `gpu_name` (or `"cpu"` if CUDA is unavailable)
- `timestamp_utc` (ISO-8601)

This way every CSV row **self-documents** the environment it was produced in.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `pydantic.ValidationError: bit_width` | `quantization.type` and `bit_width` mismatch | See validation rules in `ARCHITECTURE.md §3` |
| `OSError: ... gated repo` | Missing HF login for LLaMA | `huggingface-cli login` |
| `bitsandbytes` import error (Windows) | bnb is Linux-only | Use an fp16 config, or switch to Linux |
| `CUDA out of memory` | fp16 baseline on a small GPU | Switch to 4-bit (`bnb_nf4` / `gptq` / `awq`) |
| Row written but some columns empty | A task crashed inside the pipeline | Check stderr logs; per the error policy other tasks continue |

---

## Out of Scope

The following are **deliberately** out of scope (see `ARCHITECTURE.md §10`):

- HQQ quantization (extension point ready, not implemented yet)
- Calibration-based GPTQ/AWQ (pre-quantized HF Hub models are used instead)
- Multi-seed evaluation loop (single seed; extension point ready)
- Pareto front visualizations and statistical significance tests (analysis layer, separate notebook)
- Colab notebook integration

---

## License

TBD.
