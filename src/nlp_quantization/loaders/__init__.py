from typing import TYPE_CHECKING, Callable

from . import awq, bnb_nf4, fp16, gptq

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from ..config import ExperimentConfig

LoadFn = Callable[
    ["ExperimentConfig"], tuple["PreTrainedModel", "PreTrainedTokenizerBase"]
]

LOADER_REGISTRY: dict[str, LoadFn] = {
    "fp16": fp16.load,
    "bnb_nf4": bnb_nf4.load,
    "gptq": gptq.load,
    "awq": awq.load,
}
