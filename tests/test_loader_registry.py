from nlp_quantization.loaders import LOADER_REGISTRY


def test_registry_keys():
    assert set(LOADER_REGISTRY.keys()) == {"fp16", "bnb_nf4", "gptq", "awq"}


def test_registry_values_callable():
    for key, fn in LOADER_REGISTRY.items():
        assert callable(fn), f"{key} loader is not callable"
