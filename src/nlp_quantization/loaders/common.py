from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoTokenizer

from ..config import ExperimentConfig


def load_model_config(cfg: ExperimentConfig) -> Any:
    model_config = AutoConfig.from_pretrained(
        cfg.model.path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    normalize_rope_scaling(
        model_config,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return model_config


def load_tokenizer(cfg: ExperimentConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_rope_scaling(model_config: Any, *, trust_remote_code: bool = False) -> None:
    is_phi3 = getattr(model_config, "model_type", None) == "phi3"
    rope_scaling = getattr(model_config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        if is_phi3 and not trust_remote_code:
            _ensure_phi3_rope_parameters(model_config, None)
        return

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if rope_type is not None:
        rope_scaling.setdefault("rope_type", rope_type)
        rope_scaling.setdefault("type", rope_type)

    if not is_phi3:
        return

    if trust_remote_code:
        if rope_type == "default":
            model_config.rope_scaling = None
        return

    _ensure_phi3_rope_parameters(model_config, rope_scaling)


def _ensure_phi3_rope_parameters(model_config: Any, rope_scaling: dict[str, Any] | None) -> None:
    rope_parameters = getattr(model_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        rope_parameters = dict(rope_scaling or {})
        model_config.rope_parameters = rope_parameters

    rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type") or "default"
    rope_parameters.setdefault("rope_type", rope_type)
    rope_parameters.setdefault("type", rope_type)
    rope_parameters.setdefault("rope_theta", getattr(model_config, "rope_theta", 10000.0))
    rope_parameters.setdefault(
        "partial_rotary_factor",
        getattr(model_config, "partial_rotary_factor", 1.0),
    )
