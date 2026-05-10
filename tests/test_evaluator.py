import pytest

import nlp_quantization.evaluation.evaluator as evaluator_mod
from nlp_quantization.evaluation.evaluator import evaluate_model


METRIC_FOR_TASK = {
    "arc_challenge": "acc_norm",
    "wikitext": "word_perplexity",
    "mmlu": "acc",
}
RETURN_VALUE = {
    "arc_challenge": 0.55,
    "wikitext": 7.21,
    "mmlu": 0.62,
}


def _stub_HFLM(**kwargs):
    return object()


def _stub_simple_evaluate(*, model, tasks, num_fewshot, **kwargs):
    (task_name,) = tasks
    metric = METRIC_FOR_TASK[task_name]
    return {"results": {task_name: {f"{metric},none": RETURN_VALUE[task_name]}}}


def test_evaluate_model_returns_nine_keys(monkeypatch):
    monkeypatch.setattr(evaluator_mod, "HFLM", _stub_HFLM)
    monkeypatch.setattr(evaluator_mod, "simple_evaluate", _stub_simple_evaluate)

    out = evaluate_model(model=object(), tokenizer=object(), run_id="t")

    expected_keys = {
        "arc_challenge_acc_norm",
        "arc_challenge_peak_vram_gb",
        "arc_challenge_runtime_sec",
        "wikitext_word_perplexity",
        "wikitext_peak_vram_gb",
        "wikitext_runtime_sec",
        "mmlu_acc",
        "mmlu_peak_vram_gb",
        "mmlu_runtime_sec",
    }
    assert set(out.keys()) == expected_keys

    assert out["arc_challenge_acc_norm"] == pytest.approx(0.55)
    assert out["wikitext_word_perplexity"] == pytest.approx(7.21)
    assert out["mmlu_acc"] == pytest.approx(0.62)

    for k in (
        "arc_challenge_peak_vram_gb",
        "arc_challenge_runtime_sec",
        "wikitext_peak_vram_gb",
        "wikitext_runtime_sec",
        "mmlu_peak_vram_gb",
        "mmlu_runtime_sec",
    ):
        assert isinstance(out[k], float)
        assert out[k] >= 0.0
