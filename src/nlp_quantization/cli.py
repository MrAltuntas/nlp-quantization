from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .config import ExperimentConfig, load_config
from .run_id import generate_run_id

logger = logging.getLogger("nlp_quantization")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nlp_quantization",
        description="Run one quantization experiment defined by a YAML config.",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and emit run_id only; skip load + eval + CSV.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override cfg.output.csv_path.",
    )
    return parser.parse_args(argv)


def set_seeds(seed: int) -> None:
    import random

    import numpy
    import torch
    from transformers import set_seed as hf_set_seed

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def _collect_env() -> dict[str, str]:
    import torch
    import transformers

    try:
        import lm_eval

        lm_eval_version = getattr(lm_eval, "__version__", None)
        if lm_eval_version is None:
            from importlib.metadata import version

            lm_eval_version = version("lm-eval")
    except Exception:
        lm_eval_version = "unknown"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"

    return {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "lm_eval_version": lm_eval_version,
        "gpu_name": gpu_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _build_row(
    cfg: ExperimentConfig,
    run_id: str,
    load_metrics: dict,
    eval_metrics: dict,
    env: dict,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp_utc": env["timestamp_utc"],
        "model_path": cfg.model.path,
        "model_name": cfg.model.display_name,
        "quantization_type": cfg.quantization.type,
        "bit_width": cfg.quantization.bit_width,
        "seed": cfg.runtime.seed,
        "load_runtime_sec": load_metrics.get("load_runtime_sec"),
        "load_peak_vram_gb": load_metrics.get("load_peak_vram_gb"),
        "arc_challenge_acc_norm": eval_metrics.get("arc_challenge_acc_norm"),
        "arc_challenge_peak_vram_gb": eval_metrics.get("arc_challenge_peak_vram_gb"),
        "arc_challenge_runtime_sec": eval_metrics.get("arc_challenge_runtime_sec"),
        "wikitext_word_perplexity": eval_metrics.get("wikitext_word_perplexity"),
        "wikitext_peak_vram_gb": eval_metrics.get("wikitext_peak_vram_gb"),
        "wikitext_runtime_sec": eval_metrics.get("wikitext_runtime_sec"),
        "mmlu_acc": eval_metrics.get("mmlu_acc"),
        "mmlu_peak_vram_gb": eval_metrics.get("mmlu_peak_vram_gb"),
        "mmlu_runtime_sec": eval_metrics.get("mmlu_runtime_sec"),
        "transformers_version": env["transformers_version"],
        "torch_version": env["torch_version"],
        "lm_eval_version": env["lm_eval_version"],
        "gpu_name": env["gpu_name"],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    cfg = load_config(args.config)

    if args.csv_path is not None:
        cfg.output.csv_path = args.csv_path

    run_id = cfg.output.run_id or generate_run_id(cfg)

    if args.dry_run:
        print(run_id)
        return 0

    import torch

    from .evaluation.evaluator import evaluate_model
    from .loaders import LOADER_REGISTRY
    from .metrics import track_load
    from .results_writer import append_csv

    set_seeds(cfg.runtime.seed)

    model = None
    try:
        loader = LOADER_REGISTRY[cfg.quantization.type]
        with track_load() as load_metrics:
            model, tokenizer = loader(cfg)

        eval_metrics = evaluate_model(model, tokenizer, run_id)

        env = _collect_env()
        row = _build_row(cfg, run_id, dict(load_metrics), eval_metrics, env)
        append_csv(cfg.output.csv_path, row)
        logger.info("Wrote row to %s (run_id=%s)", cfg.output.csv_path, run_id)
        return 0

    except Exception:
        logger.exception("Pipeline failed for run_id=%s", run_id)
        return 1

    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
