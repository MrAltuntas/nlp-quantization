import pytest
import yaml
from pydantic import ValidationError

from nlp_quantization.config import ExperimentConfig, load_config


def make_data(quant_type: str, bit_width: int):
    return {
        "model": {"path": "some/repo", "display_name": "test"},
        "quantization": {"type": quant_type, "bit_width": bit_width},
    }


VALID = [
    ("fp16", 16),
    ("bnb_nf4", 4),
    ("gptq", 4),
    ("gptq", 8),
    ("awq", 4),
]

INVALID = [
    ("fp16", 4),
    ("fp16", 8),
    ("bnb_nf4", 8),
    ("bnb_nf4", 16),
    ("gptq", 16),
    ("awq", 8),
    ("awq", 16),
]


@pytest.mark.parametrize("qt,bw", VALID)
def test_valid_quant_combos(qt, bw):
    cfg = ExperimentConfig(**make_data(qt, bw))
    assert cfg.quantization.type == qt
    assert cfg.quantization.bit_width == bw


@pytest.mark.parametrize("qt,bw", INVALID)
def test_invalid_quant_combos(qt, bw):
    with pytest.raises(ValidationError):
        ExperimentConfig(**make_data(qt, bw))


def test_empty_model_path():
    data = make_data("fp16", 16)
    data["model"]["path"] = ""
    with pytest.raises(ValidationError):
        ExperimentConfig(**data)


def test_negative_seed():
    data = make_data("fp16", 16)
    data["runtime"] = {"seed": -1}
    with pytest.raises(ValidationError):
        ExperimentConfig(**data)


def test_load_config_from_yaml(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    data = {
        "model": {"path": "some/repo", "display_name": "test"},
        "quantization": {"type": "fp16", "bit_width": 16},
    }
    cfg_path.write_text(yaml.safe_dump(data))
    cfg = load_config(cfg_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.path == "some/repo"
    assert cfg.quantization.type == "fp16"
    assert cfg.runtime.seed == 1234
    assert cfg.output.csv_path == "results/results.csv"
