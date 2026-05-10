from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    path: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    trust_remote_code: bool = False


class QuantizationConfig(BaseModel):
    type: Literal["fp16", "bnb_nf4", "gptq", "awq"]
    bit_width: int


class RuntimeConfig(BaseModel):
    device: str = "cuda"
    batch_size: str | int = "auto"
    seed: int = Field(default=1234, ge=0)


class OutputConfig(BaseModel):
    csv_path: str = "results/results.csv"
    run_id: str | None = None


_BITS_FOR_TYPE: dict[str, set[int]] = {
    "fp16": {16},
    "bnb_nf4": {4},
    "gptq": {4, 8},
    "awq": {4},
}


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig
    quantization: QuantizationConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate_quant_bits(self) -> "ExperimentConfig":
        t = self.quantization.type
        b = self.quantization.bit_width
        allowed = _BITS_FOR_TYPE[t]
        if b not in allowed:
            raise ValueError(
                f"quantization.bit_width={b} not allowed for type={t!r} "
                f"(allowed: {sorted(allowed)})"
            )
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
