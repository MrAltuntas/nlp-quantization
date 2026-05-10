from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_fewshot: int
    primary_metric: str


TASKS: list[TaskSpec] = [
    TaskSpec("arc_challenge", 25, "acc_norm"),
    TaskSpec("wikitext", 0, "word_perplexity"),
    TaskSpec("mmlu", 5, "acc"),
]
