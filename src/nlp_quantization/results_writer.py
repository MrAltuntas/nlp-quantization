import csv
import fcntl
from pathlib import Path
from typing import Any

CSV_COLUMNS: list[str] = [
    "run_id",
    "timestamp_utc",
    "model_path",
    "model_name",
    "quantization_type",
    "bit_width",
    "seed",
    "load_runtime_sec",
    "load_peak_vram_gb",
    "arc_challenge_acc_norm",
    "arc_challenge_peak_vram_gb",
    "arc_challenge_runtime_sec",
    "wikitext_word_perplexity",
    "wikitext_peak_vram_gb",
    "wikitext_runtime_sec",
    "mmlu_acc",
    "mmlu_peak_vram_gb",
    "mmlu_runtime_sec",
    "transformers_version",
    "torch_version",
    "lm_eval_version",
    "gpu_name",
]


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
