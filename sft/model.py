"""
Model loading: Qwen2.5-0.5B-Instruct with LoRA (optionally quantized to 4-bit).

Provides:
  load_model_and_tokenizer() -> (model, tokenizer)

The returned model has LoRA adapters injected and is ready for Trainer.
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
    Load the base model and attach LoRA adapters.

    If quantization.enabled is true, loads in 4-bit with bitsandbytes.
    Otherwise loads in fp16 (sufficient for <=1B models on T4 16GB).

    Returns:
        model     : PeftModel ready for training
        tokenizer : AutoTokenizer with pad_token set
    """
    cfg = load_config(config_path)
    model_name = cfg["model"]["name"]
    qcfg = cfg["quantization"]
    lcfg = cfg["lora"]
    use_quant = qcfg.get("enabled", True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Load model ---
    load_kwargs = dict(
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=_dtype(cfg["model"]["torch_dtype"]),
    )

    if use_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qcfg["load_in_4bit"],
            bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=_dtype(qcfg["bnb_4bit_compute_dtype"]),
        )
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False

    if use_quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model.enable_input_require_grads()

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
