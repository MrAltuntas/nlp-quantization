# NLP Quantization Pipeline — Mimari

Bu doküman, *Impact of Quantization Methods on Large Language Model Performance Across NLP Tasks* projesinin deney pipeline'ının mimarisini tanımlar. Buradaki spec, kod yazımına başlamadan önce alınmış kararları ve modüller arası kontratları içerir.

---

## 1. Overview

Pipeline, üç açık kaynak LLM'i (Mistral-7B-v0.3, Phi-3-mini-4k, LLaMA-3.1-8B) dört quantization stratejisi (`fp16` baseline, `bnb_nf4`, `gptq`, `awq`) ile karşılaştırır ve sonuçları üç NLP görevinde değerlendirir. Her `(model × quantization × bit_width)` kombinasyonu tek bir invocation ile çalıştırılır; sonuçlar tek bir `results.csv` dosyasında biriktirilir.

Üç araştırma sorusu (RQ1–RQ3) ve karşılık gelen lm-evaluation-harness task'ları:

| RQ | Task | `task_name` | Few-shot | Primary metric |
|----|------|-------------|----------|----------------|
| RQ1 — QA accuracy | ARC-Challenge | `arc_challenge` | 25 | `acc_norm` |
| RQ2 — Quality vs efficiency | WikiText-2 | `wikitext` | 0 | `word_perplexity` |
| RQ3 — Task sensitivity | MMLU | `mmlu` | 5 | `acc` |

Quality metriklerinin yanında, her task için **peak VRAM** ve **wall-clock runtime** ölçülür. RQ2'nin efficiency boyutu (latency, memory) bu ölçümlerden türetilir.

End-to-end akış: **YAML config → model loader (quant adapter) → `evaluate_model()` → CSV row append**.

---

## 2. Repository Layout

```
nlp-quantization/
├── ARCHITECTURE.md                 # bu doküman
├── README.md
├── pyproject.toml                  # bağımlılıklar
├── .gitignore                      # results/, __pycache__/, .venv/, vb.
│
├── configs/                        # her dosya = tek deney
│   ├── mistral_fp16.yaml
│   ├── mistral_bnb_nf4.yaml
│   ├── mistral_gptq_4bit.yaml
│   ├── mistral_awq_4bit.yaml
│   ├── phi3_fp16.yaml
│   ├── phi3_bnb_nf4.yaml
│   ├── phi3_gptq_4bit.yaml
│   ├── phi3_awq_4bit.yaml
│   ├── llama3_fp16.yaml
│   ├── llama3_bnb_nf4.yaml
│   ├── llama3_gptq_4bit.yaml
│   └── llama3_awq_4bit.yaml
│
├── src/nlp_quantization/
│   ├── __init__.py
│   ├── __main__.py                 # python -m nlp_quantization …
│   ├── cli.py                      # entry point + orkestrasyon
│   ├── config.py                   # YAML → ExperimentConfig (pydantic)
│   ├── run_id.py                   # otomatik run_id üretimi
│   ├── metrics.py                  # VRAM + zamanlama context manager'ları
│   ├── results_writer.py           # append-only CSV writer
│   │
│   ├── loaders/                    # quantization adapter'ları
│   │   ├── __init__.py             # LOADER_REGISTRY: dict[str, LoadFn]
│   │   ├── fp16.py
│   │   ├── bnb_nf4.py
│   │   ├── gptq.py
│   │   └── awq.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── tasks.py                # TASKS sabit listesi
│       └── evaluator.py            # evaluate_model(...)
│
├── scripts/
│   └── run_matrix.sh               # configs/*.yaml üzerinde sırayla çalıştırır
│
└── results/
    └── results.csv                 # gitignored, append-only
```

**Tasarım prensibi**: kod modüler ve registry-pattern bazlı. Yeni bir quantization tipi eklemek için sadece `loaders/` altına dosya + registry'ye satır eklenir; pipeline'ın geri kalanına dokunulmaz.

---

## 3. Configuration Schema (YAML)

Her config dosyası **tek bir deneyi** tanımlar. Pipeline tek invocation = tek CSV satırı.

### Minimal şema

```yaml
model:
  path: TheBloke/Mistral-7B-v0.3-GPTQ
  display_name: mistral-7b-v0.3
  trust_remote_code: false

quantization:
  type: gptq
  bit_width: 4

runtime:
  device: cuda
  batch_size: auto
  seed: 1234

output:
  csv_path: results/results.csv
  run_id: null
```

### Alan açıklamaları

| Alan | Tip | Açıklama |
|------|-----|----------|
| `model.path` | str | HuggingFace repo ID veya local path. GPTQ/AWQ için pre-quantized repo. |
| `model.display_name` | str | CSV'de `model_name` olarak yazılır (kısa, okunabilir isim). |
| `model.trust_remote_code` | bool | Custom model kodu varsa `true`. Default `false`. |
| `quantization.type` | enum | `fp16` \| `bnb_nf4` \| `gptq` \| `awq` |
| `quantization.bit_width` | int | `16` (fp16), `4` (bnb_nf4 / awq), `4` veya `8` (gptq) |
| `runtime.device` | str | Default `cuda`. |
| `runtime.batch_size` | str/int | lm-eval-harness'a iletilir. `"auto"` önerilir. |
| `runtime.seed` | int | Reproducibility için. Default `1234`. |
| `output.csv_path` | str | Append edilen CSV. Default `results/results.csv`. |
| `output.run_id` | str/null | `null` ise `{display_name}__{type}__{bits}bit__{YYYYMMDD-HHMM}` formatında üretilir. |

### Validation kuralları (`config.py`, pydantic)

Pydantic validator'ları aşağıdaki kombinasyonları reddeder:

- `type == "fp16"` ⟹ `bit_width == 16`
- `type == "bnb_nf4"` ⟹ `bit_width == 4`
- `type == "gptq"` ⟹ `bit_width in (4, 8)`
- `type == "awq"` ⟹ `bit_width == 4`
- `model.path` non-empty
- `seed` ≥ 0

### Örnek config setleri

12 deney için `configs/` altında 12 dosya. Pre-quantized repo örnekleri (referans, gerçek path'ler proje başında doğrulanmalı):

| Model | fp16 | bnb_nf4 | gptq | awq |
|-------|------|---------|------|-----|
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | aynı path | `TheBloke/Mistral-7B-v0.3-GPTQ` | `solidrust/Mistral-7B-v0.3-AWQ` |
| Phi-3-mini-4k | `microsoft/Phi-3-mini-4k-instruct` | aynı path | (uygun GPTQ repo) | (uygun AWQ repo) |
| LLaMA-3.1-8B | `meta-llama/Llama-3.1-8B` | aynı path | `hugging-quants/Meta-Llama-3.1-8B-GPTQ-INT4` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |

Not: `bnb_nf4` runtime quantization olduğu için `model.path` **fp16 base modeli** gösterir; bnb load anında 4-bit'e indirir. GPTQ/AWQ için `model.path` zaten quantize edilmiş repo'yu gösterir.

---

## 4. Pipeline Flow

`cli.py` orkestrasyonun tamamı:

```
1. parse_args(--config <path>)
2. cfg = load_config(config_path)                           # YAML → ExperimentConfig
3. run_id = cfg.output.run_id or generate_run_id(cfg)
4. set_seeds(cfg.runtime.seed)                              # torch, numpy, random, transformers.set_seed
5. with track_load() as load_metrics:                       # zaman + peak VRAM
       model, tokenizer = LOADER_REGISTRY[cfg.quantization.type](cfg)
6. eval_metrics = evaluate_model(model, tokenizer, run_id)  # 3 task, flat dict
7. row = build_row(cfg, run_id, load_metrics, eval_metrics, env_metadata)
8. append_csv(cfg.output.csv_path, row)                     # header lazy bootstrap + flock
9. cleanup_gpu(model)                                       # del + empty_cache
```

**Hata politikası**: Bir task çağrısı patlarsa ilgili task kolonları boş bırakılır (NaN), diğer task'lar ve CSV append devam eder. Loader patlarsa pipeline non-zero exit ile çıkar; CSV'ye satır yazılmaz.

---

## 5. Module Responsibilities

### 5.1 `loaders/` — Quantization adapter'ları

Her loader tek imzayı uygular:

```python
def load(cfg: ExperimentConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]: ...
```

Registry (`loaders/__init__.py`):

```python
LOADER_REGISTRY: dict[str, LoadFn] = {
    "fp16":    fp16.load,
    "bnb_nf4": bnb_nf4.load,
    "gptq":    gptq.load,
    "awq":     awq.load,
}
```

| Loader | İmplementasyon özeti |
|--------|----------------------|
| `fp16.py` | `AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda")` |
| `bnb_nf4.py` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)` ile `from_pretrained(...)` |
| `gptq.py` | Pre-quantized repo: `AutoModelForCausalLM.from_pretrained(path, device_map="cuda")` (transformers'ın `auto-gptq` entegrasyonu config'i otomatik algılar) |
| `awq.py` | Pre-quantized repo: `AutoModelForCausalLM.from_pretrained(path, device_map="cuda")` (transformers'ın `autoawq` entegrasyonu) |

Tokenizer her loader'da: `AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)`. Pad token gerekiyorsa `tokenizer.pad_token = tokenizer.eos_token`.

### 5.2 `evaluation/tasks.py` — Task tanımları

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TaskSpec:
    name: str               # lm-eval task name
    num_fewshot: int
    primary_metric: str     # results dict'ten çekilen metrik

TASKS: list[TaskSpec] = [
    TaskSpec("arc_challenge", 25, "acc_norm"),
    TaskSpec("wikitext",       0, "word_perplexity"),
    TaskSpec("mmlu",           5, "acc"),
]
```

Yeni task eklemek bu listeye satır eklemekle sınırlı; CSV kolonları otomatik genişler.

### 5.3 `evaluation/evaluator.py` — Ana eval fonksiyonu

```python
def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    run_id: str,
) -> dict[str, float]:
    """
    Yüklü modeli alır, TASKS listesindeki 3 task'ı sırayla lm-eval-harness ile
    çalıştırır. Her task için peak VRAM ve wall-clock süresi ölçer.
    Flat dict döner; örn:
        {
          "arc_challenge_acc_norm":     0.55,
          "arc_challenge_peak_vram_gb": 14.2,
          "arc_challenge_runtime_sec":  320.5,
          "wikitext_word_perplexity":   7.21,
          "wikitext_peak_vram_gb":      13.8,
          "wikitext_runtime_sec":       110.3,
          "mmlu_acc":                   0.62,
          "mmlu_peak_vram_gb":          14.5,
          "mmlu_runtime_sec":           1200.7,
        }
    """
```

Her task için iç akış:

```python
import torch, time
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()

out = simple_evaluate(
    model=lm,
    tasks=[task.name],
    num_fewshot=task.num_fewshot,
)

runtime_sec  = time.perf_counter() - t0
peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
metric_value = out["results"][task.name][f"{task.primary_metric},none"]
```

`HFLM` her task çağrısı için yeniden oluşturulmaz — tek instance üç task için yeniden kullanılır (model zaten GPU'da). VRAM ölçümü her task'ın *kendi* peak'ini yakalar (reset sayesinde).

### 5.4 `metrics.py` — Ortak ölçüm primitives

Üç yardımcı:

- `track_load() -> contextmanager`: load süresi + load sonrası VRAM (`{load_runtime_sec, load_peak_vram_gb}`)
- `task_measurement(task_name) -> contextmanager`: reset → exec → ölçüm
- `bytes_to_gb(n) -> float`: `n / 1024**3`

Tüm VRAM değerleri **GB** cinsinden CSV'ye yazılır (3 ondalık).

### 5.5 `results_writer.py` — CSV writer

- **Header bootstrap**: `csv_path` yoksa veya boşsa header yazılır; varsa direkt append.
- **Dosya kilidi**: `fcntl.flock(LOCK_EX)` — paralel run'ların aynı CSV'ye güvenle yazabilmesi için.
- **Eksik kolon toleransı**: `dict[str, Any]` row'u `csv.DictWriter` ile yazılır; gelecekte yeni metrik eklenirse eski satırlar kırılmaz (mevcut header korunur, yeni run'lar için header migration ayrı bir bakım işi).

### 5.6 `run_id.py` — Otomatik run_id

Format: `{display_name}__{quant_type}__{bit_width}bit__{YYYYMMDD-HHMM}`

Örnek: `mistral-7b-v0.3__gptq__4bit__20260510-1432`

Aynı dakikada çakışma olursa son ek olarak `_{millis}` eklenir.

### 5.7 `cli.py` — Entry point

```bash
python -m nlp_quantization --config configs/mistral_gptq_4bit.yaml
```

Argümanlar:

| Flag | Açıklama |
|------|----------|
| `--config <path>` | (zorunlu) YAML config dosyası |
| `--dry-run` | Config validation + run_id üretimi yapar, model yüklemez |
| `--csv-path <path>` | Config'deki `output.csv_path`'i override eder |

---

## 6. Output Schema (`results.csv`)

Sabit kolon sırası:

```
run_id,
timestamp_utc,
model_path,
model_name,
quantization_type,
bit_width,
seed,
load_runtime_sec,
load_peak_vram_gb,
arc_challenge_acc_norm,
arc_challenge_peak_vram_gb,
arc_challenge_runtime_sec,
wikitext_word_perplexity,
wikitext_peak_vram_gb,
wikitext_runtime_sec,
mmlu_acc,
mmlu_peak_vram_gb,
mmlu_runtime_sec,
transformers_version,
torch_version,
lm_eval_version,
gpu_name
```

**Bir satır = bir deney = bir `(model × quantization × bit_width)` kombinasyonu.**

Tam matris çalıştırıldığında: `3 model × 4 quantization = 12 satır`.

CSV doğrudan pandas'a yüklenip RQ analizleri yapılabilir:

- **RQ1**: `arc_challenge_acc_norm` kolonunu `quantization_type` ile gruplandır
- **RQ2**: `wikitext_word_perplexity` × `wikitext_peak_vram_gb` Pareto front
- **RQ3**: Üç task metriğinin fp16 baseline'a göre yüzde değişimi → task hassasiyeti

---

## 7. Execution

### Tek deney

```bash
python -m nlp_quantization --config configs/mistral_gptq_4bit.yaml
```

### Tüm matris (helper script)

```bash
bash scripts/run_matrix.sh
```

`run_matrix.sh` iskelet mantığı:

```bash
#!/usr/bin/env bash
set -euo pipefail
for cfg in configs/*.yaml; do
    echo "=== Running: $cfg ==="
    python -m nlp_quantization --config "$cfg" || echo "FAILED: $cfg"
done
```

Her invocation `results/results.csv`'ye bir satır appendler. Bir deney çökerse diğerleri etkilenmez.

---

## 8. Reproducibility & Metadata

Her run aşağıdaki metadata'yı CSV'ye yazar:

| Kolon | Kaynak |
|-------|--------|
| `seed` | `cfg.runtime.seed` (default `1234`) |
| `transformers_version` | `transformers.__version__` |
| `torch_version` | `torch.__version__` |
| `lm_eval_version` | `lm_eval.__version__` |
| `gpu_name` | `torch.cuda.get_device_name(0)` |
| `timestamp_utc` | `datetime.now(timezone.utc).isoformat()` |

Seed kurulumu (`set_seeds`):

```python
import random, numpy, torch
from transformers import set_seed as hf_set_seed

def set_seeds(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
```

---

## 9. Extension Points

Mimaride bilinçli bırakılan genişleme noktaları:

| İhtiyaç | Genişleme yolu |
|---------|----------------|
| Yeni quantization yöntemi (HQQ, FP8, vb.) | `loaders/<yeni>.py` + `LOADER_REGISTRY`'ye satır + config validation kuralı |
| Yeni task / RQ | `evaluation/tasks.py` içindeki `TASKS` listesine `TaskSpec` ekle; CSV kolonları otomatik genişler |
| Multi-seed loop | `cli.py`'a `--seeds 1234,5678,9012` flag'i; her seed ayrı CSV satırı (run_id farklı) |
| Cluster / Slurm execution | `scripts/` altına yeni runner; pipeline tek-invocation tasarımı uyumlu |
| Custom calibration ile GPTQ/AWQ | Yeni loader (`loaders/gptq_calibrated.py`) + config'e `calibration` bloğu |

---

## 10. Out of Scope (şimdilik)

Mimari kapsamı dışında bırakılan ve ileride eklenebilecek konular:

- **HQQ quantization** — proje önerisinde vardı ama kapsamdan çıkarıldı
- **Colab notebook entegrasyonu** — Colab tarafı şimdilik umursanmıyor; runner saf Python CLI
- **Calibration-based GPTQ/AWQ** — pre-quantized HF Hub modelleri kullanılacak
- **Pareto front görselleştirmeleri** — analiz katmanında, ayrı notebook'ta
- **Statistical significance testleri** (paired t-test, Wilcoxon) — analiz katmanında, CSV üzerinden pandas/scipy ile
- **Multi-seed evaluation loop** — şimdilik tek-seed; mimari extension point'i hazır

---

## 11. Bağımlılıklar (referans)

`pyproject.toml`'a girecek minimum bağımlılıklar:

```
torch
transformers
accelerate
bitsandbytes
auto-gptq
autoawq
lm-eval
pyyaml
pydantic
```

Sürüm pinleri ilk implementasyonda belirlenir ve `pyproject.toml`'da sabitlenir.
