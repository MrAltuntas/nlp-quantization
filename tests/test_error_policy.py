import nlp_quantization.evaluation.evaluator as evaluator_mod
from nlp_quantization.evaluation.evaluator import evaluate_model


METRIC_FOR_TASK = {
    "arc_challenge": "acc_norm",
    "wikitext": "word_perplexity",
    "mmlu": "acc",
}


def _stub_HFLM(**kwargs):
    return object()


def _stub_with_wikitext_failure(*, model, tasks, num_fewshot, **kwargs):
    (task_name,) = tasks
    if task_name == "wikitext":
        raise RuntimeError("boom")
    return {"results": {task_name: {f"{METRIC_FOR_TASK[task_name]},none": 0.5}}}


def test_failed_task_columns_are_none(monkeypatch):
    monkeypatch.setattr(evaluator_mod, "HFLM", _stub_HFLM)
    monkeypatch.setattr(evaluator_mod, "simple_evaluate", _stub_with_wikitext_failure)

    out = evaluate_model(model=object(), tokenizer=object(), run_id="t")

    assert out["wikitext_word_perplexity"] is None
    assert out["wikitext_peak_vram_gb"] is None
    assert out["wikitext_runtime_sec"] is None

    assert isinstance(out["arc_challenge_acc_norm"], float)
    assert isinstance(out["mmlu_acc"], float)
