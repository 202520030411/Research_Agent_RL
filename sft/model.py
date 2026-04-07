"""
Model loading: Qwen2.5-0.5B-Instruct with 4-bit QLoRA.

Provides:
  load_model_and_tokenizer() -> (model, tokenizer)

The returned model has LoRA adapters injected and is ready for SFTTrainer.
"""

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_model_and_tokenizer(
    config_path: str = "config.yaml",
) -> tuple:
    """
    Load the quantized base model and attach LoRA adapters.

    Returns:
        model     : PeftModel ready for gradient-checkpointed training
        tokenizer : AutoTokenizer with pad_token set
    """
    cfg = load_config(config_path)
    model_name = cfg["model"]["name"]
    qcfg = cfg["quantization"]
    lcfg = cfg["lora"]

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=_dtype(qcfg["bnb_4bit_compute_dtype"]),
    )

    # --- Base model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=_dtype(cfg["model"]["torch_dtype"]),
    )

    # Required before applying LoRA to a quantized model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        target_modules=lcfg["target_modules"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
