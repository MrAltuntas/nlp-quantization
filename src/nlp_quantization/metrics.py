import time
from contextlib import contextmanager
from typing import Iterator

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def bytes_to_gb(n: int) -> float:
    return n / 1024**3


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


@contextmanager
def track_load() -> Iterator[dict[str, float]]:
    metrics: dict[str, float] = {"load_runtime_sec": 0.0, "load_peak_vram_gb": 0.0}
    if _cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics["load_runtime_sec"] = time.perf_counter() - t0
        if _cuda_available():
            metrics["load_peak_vram_gb"] = bytes_to_gb(torch.cuda.max_memory_allocated())


@contextmanager
def task_measurement(task_name: str) -> Iterator[dict[str, float]]:
    metrics: dict[str, float] = {"peak_vram_gb": 0.0, "runtime_sec": 0.0}
    if _cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics["runtime_sec"] = time.perf_counter() - t0
        if _cuda_available():
            metrics["peak_vram_gb"] = bytes_to_gb(torch.cuda.max_memory_allocated())
