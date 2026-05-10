from dataclasses import FrozenInstanceError

import pytest

from nlp_quantization.evaluation.tasks import TASKS, TaskSpec


def test_tasks_count():
    assert len(TASKS) == 3


def test_tasks_names():
    assert {t.name for t in TASKS} == {"arc_challenge", "wikitext", "mmlu"}


def test_tasks_fewshot():
    assert TASKS[0].num_fewshot == 25
    assert TASKS[1].num_fewshot == 0
    assert TASKS[2].num_fewshot == 5


def test_tasks_primary_metric():
    by_name = {t.name: t for t in TASKS}
    assert by_name["arc_challenge"].primary_metric == "acc_norm"
    assert by_name["wikitext"].primary_metric == "word_perplexity"
    assert by_name["mmlu"].primary_metric == "acc"


def test_taskspec_is_frozen():
    spec = TaskSpec("x", 0, "y")
    with pytest.raises(FrozenInstanceError):
        spec.name = "z"
