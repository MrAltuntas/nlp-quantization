from types import SimpleNamespace

from nlp_quantization.loaders.common import normalize_rope_scaling


def test_normalize_rope_scaling_copies_rope_type_to_legacy_type_key():
    cfg = SimpleNamespace(rope_scaling={"rope_theta": 10000.0, "rope_type": "default"})

    normalize_rope_scaling(cfg)

    assert cfg.rope_scaling["type"] == "default"


def test_normalize_rope_scaling_keeps_default_phi3_rope_parameters():
    cfg = SimpleNamespace(
        model_type="phi3",
        rope_scaling={"rope_theta": 10000.0, "rope_type": "default"},
    )

    normalize_rope_scaling(cfg)

    assert cfg.rope_scaling["rope_type"] == "default"
    assert cfg.rope_scaling["type"] == "default"
    assert cfg.rope_parameters["rope_type"] == "default"
    assert cfg.rope_parameters["rope_theta"] == 10000.0
    assert cfg.rope_parameters["partial_rotary_factor"] == 1.0


def test_normalize_rope_scaling_disables_default_phi3_remote_rope_scaling():
    cfg = SimpleNamespace(
        model_type="phi3",
        rope_scaling={"rope_theta": 10000.0, "rope_type": "default"},
    )

    normalize_rope_scaling(cfg, trust_remote_code=True)

    assert cfg.rope_scaling is None
    assert not hasattr(cfg, "rope_parameters")


def test_normalize_rope_scaling_restores_missing_phi3_rope_parameters():
    cfg = SimpleNamespace(model_type="phi3", rope_scaling=None, rope_parameters=None)

    normalize_rope_scaling(cfg)

    assert cfg.rope_parameters == {
        "rope_type": "default",
        "type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    }


def test_normalize_rope_scaling_keeps_existing_type_key():
    cfg = SimpleNamespace(rope_scaling={"rope_type": "longrope", "type": "custom"})

    normalize_rope_scaling(cfg)

    assert cfg.rope_scaling["type"] == "custom"


def test_normalize_rope_scaling_ignores_missing_rope_scaling():
    cfg = SimpleNamespace()

    normalize_rope_scaling(cfg)

    assert not hasattr(cfg, "rope_scaling")
