from pathlib import Path

import pytest

from nlp_quantization.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))


def test_twelve_config_files():
    assert len(CONFIG_FILES) == 12, f"expected 12 configs, found {len(CONFIG_FILES)}"


@pytest.mark.parametrize(
    "config_path", CONFIG_FILES, ids=[p.name for p in CONFIG_FILES]
)
def test_config_parses(config_path):
    cfg = load_config(config_path)
    assert cfg is not None


@pytest.mark.parametrize(
    "config_path", CONFIG_FILES, ids=[p.name for p in CONFIG_FILES]
)
def test_config_filename_matches_content(config_path):
    cfg = load_config(config_path)
    stem = config_path.stem

    if stem.endswith("_fp16"):
        assert cfg.quantization.type == "fp16"
        assert cfg.quantization.bit_width == 16
    elif stem.endswith("_bnb_nf4"):
        assert cfg.quantization.type == "bnb_nf4"
        assert cfg.quantization.bit_width == 4
    elif stem.endswith("_gptq_4bit"):
        assert cfg.quantization.type == "gptq"
        assert cfg.quantization.bit_width == 4
    elif stem.endswith("_awq_4bit"):
        assert cfg.quantization.type == "awq"
        assert cfg.quantization.bit_width == 4
    else:
        pytest.fail(f"unknown config filename pattern: {stem}")


def test_phi3_base_configs_use_builtin_transformers_implementation():
    for name in ["phi3_fp16.yaml", "phi3_bnb_nf4.yaml"]:
        cfg = load_config(CONFIGS_DIR / name)
        assert cfg.model.trust_remote_code is False
