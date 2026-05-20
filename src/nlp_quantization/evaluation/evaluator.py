from __future__ import annotations

import logging
from typing import Any

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from ..metrics import task_measurement
from .tasks import TASKS

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    run_id: str,
) -> dict[str, float | None]:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

    results: dict[str, float | None] = {}

    for task in TASKS:
        metric_key = f"{task.name}_{task.primary_metric}"
        vram_key = f"{task.name}_peak_vram_gb"
        time_key = f"{task.name}_runtime_sec"

        try:
            with task_measurement(task.name) as m:
                out = simple_evaluate(
                    model=lm,
                    tasks=[task.name],
                    num_fewshot=task.num_fewshot,
                )
                metric_value = out["results"][task.name][f"{task.primary_metric},none"]

            results[metric_key] = float(metric_value)
            results[vram_key] = float(m["peak_vram_gb"])
            results[time_key] = float(m["runtime_sec"])

        except Exception:
            logger.exception(
                "Task %s failed for run_id=%s; recording None for its 3 columns.",
                task.name,
                run_id,
            )
            results[metric_key] = None
            results[vram_key] = None
            results[time_key] = None

    return results
