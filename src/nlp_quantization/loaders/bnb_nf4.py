import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from ..config import ExperimentConfig
from .common import load_model_config, load_tokenizer


def load(cfg: ExperimentConfig):
    model_config = load_model_config(cfg)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        config=model_config,
        quantization_config=quant_config,
        device_map="cuda",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer = load_tokenizer(cfg)
    return model, tokenizer
