import torch
from transformers import AutoModelForCausalLM

from ..config import ExperimentConfig
from .common import load_model_config, load_tokenizer


def load(cfg: ExperimentConfig):
    model_config = load_model_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        config=model_config,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer = load_tokenizer(cfg)
    return model, tokenizer
