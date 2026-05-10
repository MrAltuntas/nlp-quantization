import importlib

import pytest


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize(
    "module_name",
    [
        "nlp_quantization.loaders.bnb_nf4",
        "nlp_quantization.loaders.gptq",
        "nlp_quantization.loaders.awq",
    ],
)
def test_loader_module_importable(module_name):
    importlib.import_module(module_name)


@pytest.mark.skipif(not _has_cuda(), reason="CUDA gerekli")
def test_fp16_loader_with_tiny_model():
    from nlp_quantization.config import ExperimentConfig
    from nlp_quantization.loaders import LOADER_REGISTRY

    cfg = ExperimentConfig(
        model={
            "path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "display_name": "tiny-llama",
        },
        quantization={"type": "fp16", "bit_width": 16},
    )
    model, tokenizer = LOADER_REGISTRY["fp16"](cfg)
    assert model is not None
    assert tokenizer.pad_token is not None
