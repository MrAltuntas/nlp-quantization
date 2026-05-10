import re
from datetime import datetime

import pytest

from nlp_quantization import run_id as run_id_module
from nlp_quantization.config import ExperimentConfig

REGEX = re.compile(
    r"^[\w\-\.]+__(fp16|bnb_nf4|gptq|awq)__\d+bit__\d{8}-\d{4}(_\d+)?$"
)


def make_cfg():
    return ExperimentConfig(
        model={"path": "some/repo", "display_name": "test-model"},
        quantization={"type": "fp16", "bit_width": 16},
    )


@pytest.fixture(autouse=True)
def _clear_generated():
    run_id_module._GENERATED.clear()
    yield
    run_id_module._GENERATED.clear()


class FakeDateTime:
    _fixed = datetime(2026, 5, 10, 14, 32, 0, 123456)

    @classmethod
    def now(cls):
        return cls._fixed


def test_generate_run_id_format(monkeypatch):
    monkeypatch.setattr(run_id_module, "datetime", FakeDateTime)
    cfg = make_cfg()
    rid = run_id_module.generate_run_id(cfg)
    assert REGEX.match(rid), f"run_id does not match regex: {rid}"
    assert rid == "test-model__fp16__16bit__20260510-1432"


def test_generate_run_id_collision(monkeypatch):
    monkeypatch.setattr(run_id_module, "datetime", FakeDateTime)
    cfg = make_cfg()
    id1 = run_id_module.generate_run_id(cfg)
    id2 = run_id_module.generate_run_id(cfg)
    assert id1 != id2
    assert REGEX.match(id2)
    assert "_" in id2.rsplit("__", 1)[-1]
