import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import ExperimentConfig


def load(cfg: ExperimentConfig):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        quantization_config=quant_config,
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
