# NLP Quantization Pipeline — Implementation Guide

Bu doküman, [`ARCHITECTURE.md`](ARCHITECTURE.md) içinde tanımlanan pipeline'ın **3 sıralı phase** içinde nasıl inşa edileceğini anlatır. Hedef okuyucu, bu rehberi adım adım takip eden bir kod ajanıdır (insan veya AI). Mimari kararlar `ARCHITECTURE.md`'de; bu doküman yalnızca **hangi sıra ile, hangi testlerle** inşa edileceğini söyler.

---

## Genel Kurallar

1. **Phase'ler bağımlıdır**. Phase 1 testleri yeşil olmadan Phase 2'ye **geçilmez**. Aynı şekilde Phase 2 → Phase 3.
2. Her phase sonunda yazdığın testleri `pytest tests/ -v` komutuyla çalıştır. Hepsi yeşil olmalı.
3. `ARCHITECTURE.md` ile bu doküman çelişirse `ARCHITECTURE.md` doğrudur — orası spec, burası yürütme planı.
4. Yeni kavram, yeni alan veya yeni dosya **icat etme**. Sadece `ARCHITECTURE.md`'de geçen modülleri yaz.
5. Hızlı geri bildirim için: GPU gerektiren testler `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gerekli")` ile koru. Mock'lanabilen yerler mock'lansın; gerçek 7B model **hiçbir testte** indirilmez.
6. Test stratejisi: Phase 1 saf birim testleri, Phase 2 light smoke + tiny model (`hf-internal-testing/tiny-random-LlamaForCausalLM`), Phase 3 monkeypatch'li mock + dry-run.

---

## Phase 1 — Foundation (GPU-bağımsız çekirdek)

### Scope

`ARCHITECTURE.md` §3 (config schema), §5.2 (tasks), §5.4 (metrics), §5.5 (results writer), §5.6 (run_id) modüllerinin saf-Python kısmı. Bu phase'de **torch import edilmez**, **transformers çağrılmaz**, **GPU'ya dokunulmaz**.

### Dosyalar

Oluşturulacak:

- `pyproject.toml` — `[project]` ve `[project.optional-dependencies] dev`. Runtime bağımlılıklar (`pyyaml`, `pydantic`); dev bağımlılıkları (`pytest`). Torch/transformers/lm-eval Phase 2 ve 3'te eklenir.
- `.gitignore` — `results/`, `__pycache__/`, `.venv/`, `*.egg-info/`, `.pytest_cache/`, `.idea/`
- `src/nlp_quantization/__init__.py` (boş veya `__version__`)
- `src/nlp_quantization/config.py` — Pydantic `ExperimentConfig`, alt modeller (`ModelConfig`, `QuantizationConfig`, `RuntimeConfig`, `OutputConfig`), `load_config(path: str) -> ExperimentConfig` fonksiyonu, §3'teki 4 validation kuralı.
- `src/nlp_quantization/run_id.py` — `generate_run_id(cfg) -> str`, format `{display_name}__{type}__{bits}bit__{YYYYMMDD-HHMM}`. Aynı dakikada çakışma → `_{millis}` suffix.
- `src/nlp_quantization/metrics.py` — `track_load()` ve `task_measurement(task_name)` context manager'ları, `bytes_to_gb(n) -> float`. GPU yoksa VRAM 0.0 döndürsün.
- `src/nlp_quantization/results_writer.py` — `append_csv(path: str, row: dict) -> None`. Header lazy bootstrap, `fcntl.flock(LOCK_EX)`, `csv.DictWriter(extrasaction="ignore")`. CSV kolon sırası §6'daki sabit liste.
- `src/nlp_quantization/evaluation/__init__.py`
- `src/nlp_quantization/evaluation/tasks.py` — `TaskSpec` frozen dataclass, `TASKS` listesi (3 eleman, §5.2'deki tablo).
- `tests/__init__.py`, `tests/conftest.py` (gerekirse), `tests/test_config.py`, `tests/test_run_id.py`, `tests/test_results_writer.py`, `tests/test_tasks.py`

### Görevler

1. `pyproject.toml` ve `.gitignore` yaz. `pip install -e ".[dev]"` çalışmalı.
2. `config.py`: Pydantic modelleri + `load_config()`. Validator'larda §3'teki 4 quantization kuralını ayrı ayrı yazıyorsun (model-level validator).
3. `run_id.py`: deterministik format + millis fallback.
4. `metrics.py`: 2 context manager, `bytes_to_gb`. CUDA yoksa `track_load()` `peak_vram_gb=0.0` döndürür.
5. `results_writer.py`: header bootstrap (dosya yoksa veya boşsa header yaz), append, lock.
6. `evaluation/tasks.py`: `TaskSpec` + `TASKS = [TaskSpec("arc_challenge", 25, "acc_norm"), TaskSpec("wikitext", 0, "word_perplexity"), TaskSpec("mmlu", 5, "acc")]`.
7. Aşağıdaki test dosyalarını yaz ve çalıştır.

### Tests

`tests/test_config.py`:
- `pytest.parametrize` ile geçerli kombinasyonlar — kabul: `(fp16, 16)`, `(bnb_nf4, 4)`, `(gptq, 4)`, `(gptq, 8)`, `(awq, 4)`. Reddedilenler: `(fp16, 4)`, `(fp16, 8)`, `(bnb_nf4, 8)`, `(bnb_nf4, 16)`, `(gptq, 16)`, `(awq, 8)`, `(awq, 16)`. Reddedilenlerde `pydantic.ValidationError` beklenir.
- `model.path` boş string → `ValidationError`.
- `seed = -1` → `ValidationError`.
- Geçerli minimal YAML'ı tmp_path'e yazıp `load_config()` ile parse et — `ExperimentConfig` instance dönsün.

`tests/test_run_id.py`:
- `generate_run_id(cfg)` çıktısı şu regex'le eşleşmeli: `r"^[\w\-\.]+__(fp16|bnb_nf4|gptq|awq)__\d+bit__\d{8}-\d{4}(_\d+)?$"`.
- Aynı `cfg` ile arka arkaya iki çağrı (aynı dakika içinde) → ikinci ID'nin sonunda `_<millis>` suffix var.

`tests/test_results_writer.py`:
- `tmp_path/results.csv` yoksa: `append_csv(...)` ilk çağrıda header + 1 satır yazar.
- Aynı path'e ikinci `append_csv` çağrısı → header tekrar yazılmaz, satır sayısı 2 olur.
- Row dict'inde header'da olmayan ekstra anahtar → `extrasaction="ignore"` davranışı (hata fırlatmaz).
- Row dict'inde eksik anahtar → o kolon boş string yazılır.

`tests/test_tasks.py`:
- `len(TASKS) == 3`, isimler set olarak `{"arc_challenge", "wikitext", "mmlu"}`.
- `TASKS[0].num_fewshot == 25`, `TASKS[1].num_fewshot == 0`, `TASKS[2].num_fewshot == 5`.
- `TaskSpec` frozen — alanına atama denemesi `dataclasses.FrozenInstanceError` fırlatır.

### Çalıştırma

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Definition of Done

- Yukarıdaki 4 test dosyasındaki tüm testler yeşil.
- `python -c "from nlp_quantization.config import ExperimentConfig; from nlp_quantization.evaluation.tasks import TASKS; print(len(TASKS))"` çıktısı `3`.
- `python -c "from nlp_quantization.results_writer import append_csv"` import hatası yok.

---

## Phase 2 — Loaders + Configs (quantization adapter'ları)

### Scope

`ARCHITECTURE.md` §5.1 (loaders) ve §3'teki 12 deney config'i. Bu phase'de torch/transformers/bitsandbytes/auto-gptq/autoawq bağımlılıkları eklenir, ama testler **gerçek 7B model indirmez** — sadece tiny model veya import smoke.

### Dosyalar

Oluşturulacak:

- `pyproject.toml` güncelle: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `auto-gptq`, `autoawq` ekle.
- `src/nlp_quantization/loaders/__init__.py` — `LOADER_REGISTRY: dict[str, LoadFn]`, `LoadFn` tip alias.
- `src/nlp_quantization/loaders/fp16.py` — `def load(cfg) -> tuple[PreTrainedModel, PreTrainedTokenizer]`. `AutoModelForCausalLM.from_pretrained(cfg.model.path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=cfg.model.trust_remote_code)`.
- `src/nlp_quantization/loaders/bnb_nf4.py` — `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`.
- `src/nlp_quantization/loaders/gptq.py` — pre-quantized repo'dan `from_pretrained(..., device_map="cuda")`.
- `src/nlp_quantization/loaders/awq.py` — pre-quantized repo'dan `from_pretrained(..., device_map="cuda")`.
- `configs/mistral_fp16.yaml`, `configs/mistral_bnb_nf4.yaml`, `configs/mistral_gptq_4bit.yaml`, `configs/mistral_awq_4bit.yaml`
- `configs/phi3_fp16.yaml`, `configs/phi3_bnb_nf4.yaml`, `configs/phi3_gptq_4bit.yaml`, `configs/phi3_awq_4bit.yaml`
- `configs/llama3_fp16.yaml`, `configs/llama3_bnb_nf4.yaml`, `configs/llama3_gptq_4bit.yaml`, `configs/llama3_awq_4bit.yaml`
- `tests/test_loader_registry.py`, `tests/test_configs.py`, `tests/test_loaders_smoke.py`

Her loader'da tokenizer: `AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)`. `tokenizer.pad_token is None` ise `tokenizer.pad_token = tokenizer.eos_token`.

### Görevler

1. `pyproject.toml`'da bağımlılıkları güncelle, `pip install -e ".[dev]"` yeniden çalıştır.
2. `loaders/__init__.py`'da `LOADER_REGISTRY = {"fp16": fp16.load, "bnb_nf4": bnb_nf4.load, "gptq": gptq.load, "awq": awq.load}`.
3. 4 loader dosyasını yaz; ortak `pad_token` mantığını her birinde tekrarla (3 satır, abstrakte etmeye değmez).
4. 12 YAML config dosyasını §3'teki örnek tabloya göre doldur. Pre-quantized repo'lar için ARCHITECTURE.md §3'teki referans tablo ilk değer olarak kullanılabilir; `display_name` kısa ve okunabilir olsun.
5. Aşağıdaki test dosyalarını yaz ve çalıştır.

### Tests

`tests/test_loader_registry.py`:
- `from nlp_quantization.loaders import LOADER_REGISTRY` import edilebilir.
- Anahtarlar tam olarak `{"fp16", "bnb_nf4", "gptq", "awq"}`.
- Her değer `callable`.

`tests/test_configs.py`:
- `glob("configs/*.yaml")` 12 dosya bulur.
- Her dosyayı `load_config(path)` ile parse et — hiçbiri `ValidationError` fırlatmamalı.
- Her config için `quantization.type` ile dosya adı uyumlu (örn. `mistral_gptq_4bit.yaml` → `type=="gptq"` ve `bit_width==4`).

`tests/test_loaders_smoke.py`:
- `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gerekli")` ile korunan tek bir test:
  - Tiny config oluştur: `model.path = "hf-internal-testing/tiny-random-LlamaForCausalLM"`, `quant.type="fp16"`, `bit_width=16`.
  - `model, tok = LOADER_REGISTRY["fp16"](cfg)` çağır.
  - `model` `None` değil; `tok.pad_token is not None`.
- CUDA gerektirmeyen import smoke: `bnb_nf4`, `gptq`, `awq` modüllerinin **import edilebildiğini** doğrula (`importlib.import_module(...)` raise etmesin). Gerçek loading bu phase'de denenmez.

### Çalıştırma

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Definition of Done

- Phase 1 + Phase 2 tüm testleri yeşil.
- `configs/` altında **tam 12** YAML dosyası, hepsi parse + validate.
- `python -c "from nlp_quantization.loaders import LOADER_REGISTRY; print(sorted(LOADER_REGISTRY))"` çıktısı `['awq', 'bnb_nf4', 'fp16', 'gptq']`.

---

## Phase 3 — Evaluation + CLI (end-to-end orkestrasyon)

### Scope

`ARCHITECTURE.md` §4 (pipeline flow), §5.3 (evaluator), §5.7 (CLI), §7 (execution), §8 (reproducibility). Bu phase'de `lm-eval` bağımlılığı eklenir; orkestrasyon, hata politikası ve metadata yazımı tamamlanır.

### Dosyalar

Oluşturulacak:

- `pyproject.toml` güncelle: `lm-eval` ekle.
- `src/nlp_quantization/__main__.py` — `from .cli import main; main()`.
- `src/nlp_quantization/cli.py` — argparse: `--config` (zorunlu), `--dry-run`, `--csv-path`. §4'teki 9 adımlı orkestrasyon. `set_seeds()` fonksiyonu burada veya ayrı `seeds.py`'de (ek dosya istemiyorsan `cli.py` içinde tut).
- `src/nlp_quantization/evaluation/evaluator.py` — `evaluate_model(model, tokenizer, run_id) -> dict[str, float | None]`. §5.3'teki iç akış (HFLM tek instance, 3 task döngüsü, her task öncesi `reset_peak_memory_stats`). Bir task `Exception` raise ederse: o task'ın kolonlarını `None` yap, diğerlerine devam et, log bas.
- `scripts/run_matrix.sh` — §7'deki iskelet (`set -euo pipefail`, for döngüsü, hata toleranslı `|| echo "FAILED: $cfg"`).
- `tests/test_cli_dry_run.py`, `tests/test_evaluator.py`, `tests/test_error_policy.py`, `tests/test_set_seeds.py`

### Görevler

1. `pyproject.toml`'a `lm-eval` ekle, `pip install -e ".[dev]"` çalıştır.
2. `set_seeds(seed)` fonksiyonunu yaz (§8'deki 5 satır).
3. `evaluator.py`'da `evaluate_model()` yaz. Flat dict çıktısı: her task için `<task>_<primary>`, `<task>_peak_vram_gb`, `<task>_runtime_sec`. Hata politikası: try/except, başarısız task'ın 3 kolonu da `None`.
4. `cli.py`'da §4'teki 9 adımı uygula. `env_metadata`: `transformers.__version__`, `torch.__version__`, `lm_eval.__version__`, `torch.cuda.get_device_name(0)` (CUDA yoksa `"cpu"`), `datetime.now(timezone.utc).isoformat()`. `--dry-run`: model yüklenmez, run_id stdout'a basılır, exit 0, CSV'ye satır yazılmaz.
5. `__main__.py` ve `scripts/run_matrix.sh` yaz; `chmod +x scripts/run_matrix.sh`.
6. Aşağıdaki test dosyalarını yaz ve çalıştır.

### Tests

`tests/test_cli_dry_run.py`:
- `subprocess.run(["python", "-m", "nlp_quantization", "--config", "configs/mistral_fp16.yaml", "--dry-run"], capture_output=True)` çağır.
- `returncode == 0`, `stdout` içinde üretilen run_id (`__fp16__16bit__` substring'i ile) görünür.
- `tmp_path` yönlendirmesi ile CSV yazılmadığını doğrula (önce `--csv-path tmp_path/x.csv`, sonra dosya var mı kontrol et — olmamalı).

`tests/test_evaluator.py`:
- `monkeypatch` ile `lm_eval.simple_evaluate`'ı stub'la: stub her task için sahte sonuç döndürür (`{"results": {task_name: {"acc_norm,none": 0.5, "word_perplexity,none": 7.0, "acc,none": 0.6}}}` gibi — task'a göre uygun anahtar).
- `evaluate_model(mock_model, mock_tokenizer, "test_run")` çağır.
- Çıktı dict'inde **9 anahtar**: `arc_challenge_acc_norm`, `arc_challenge_peak_vram_gb`, `arc_challenge_runtime_sec`, `wikitext_word_perplexity`, `wikitext_peak_vram_gb`, `wikitext_runtime_sec`, `mmlu_acc`, `mmlu_peak_vram_gb`, `mmlu_runtime_sec`.
- `peak_vram_gb` ve `runtime_sec` değerleri `float` ve `>= 0`.

`tests/test_error_policy.py`:
- `monkeypatch`'le stub `simple_evaluate` — `wikitext` çağrısında `RuntimeError("boom")` raise et, diğer iki task'ta normal sonuç dön.
- `evaluate_model(...)` çağrısı **raise etmemeli**.
- Çıktıda `wikitext_word_perplexity is None`, `wikitext_peak_vram_gb is None`, `wikitext_runtime_sec is None`.
- `arc_challenge_acc_norm` ve `mmlu_acc` normal `float` değerli.

`tests/test_set_seeds.py`:
- `set_seeds(1234)`; `import torch, numpy`; `t1 = torch.randn(3); n1 = numpy.random.rand(3)`.
- `set_seeds(1234)` tekrar; `t2 = torch.randn(3); n2 = numpy.random.rand(3)`.
- `torch.equal(t1, t2)` ve `numpy.allclose(n1, n2)` — deterministik.

### Çalıştırma

```bash
pip install -e ".[dev]"
pytest tests/ -v
python -m nlp_quantization --config configs/mistral_fp16.yaml --dry-run
```

### Definition of Done

- Phase 1 + 2 + 3 tüm testler yeşil (`pytest tests/ -v` toplamda hata yok).
- `python -m nlp_quantization --config configs/<herhangi>.yaml --dry-run` exit kodu 0.
- `bash scripts/run_matrix.sh` (sadece syntax — `bash -n scripts/run_matrix.sh` hatasız).
- `results/results.csv` formatı §6'daki sabit kolon sırasıyla yazılıyor (gerçek bir GPU'lu run gerektirmez; dry-run + manuel `append_csv` testiyle doğrulandı).

---

## Phase Sonrası — Manuel Doğrulama (opsiyonel, GPU gerekli)

Üç phase de tamamlandıktan sonra gerçek bir uçtan-uca run:

```bash
python -m nlp_quantization --config configs/mistral_fp16.yaml
cat results/results.csv
```

Beklenen: tek satır CSV, §6'daki tüm kolonlar dolu, NaN olmayan task metrikleri.

Tüm matris (12 deney, GPU'da saatlerce sürer):

```bash
bash scripts/run_matrix.sh
```

Bu adımlar `implementation.md` kapsamında **otomatik test edilmez** — manuel çalıştırma + sonuç incelemesi.
