from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig


def load(cfg: ExperimentConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        device_map="cuda",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
