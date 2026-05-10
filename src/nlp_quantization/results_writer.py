import csv
import os
import time
from pathlib import Path
from typing import Any

if os.name == "nt":
    import msvcrt

    def _lock(f) -> None:
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)

    def _unlock(f) -> None:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

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
    with open(p, "a+", newline="") as f:
        _lock(f)
        try:
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        finally:
            _unlock(f)
