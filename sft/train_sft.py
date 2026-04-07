"""
SFT training entry point.

Usage (from repo root):
  # Step 1: Build the dataset (only needed once)
  python data/prepare_sft_dataset.py

  # Step 2: Train
  python sft/train_sft.py
  python sft/train_sft.py --config config.yaml   # explicit config path

The script:
  1. Loads config.yaml
  2. Loads the JSONL traces produced by prepare_sft_dataset.py
  3. Loads Qwen-0.6B-Instruct + QLoRA via sft/model.py
  4. Runs TRL SFTTrainer with instruction masking
  5. Saves the LoRA adapter to checkpoints/qwen-sft/final
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import yaml

# Allow running from both repo root and sft/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sft_dataset import SFTTraceDataset, collate_fn
from sft.model import load_model_and_tokenizer

from transformers import Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for research agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_dataset_exists(cfg: dict) -> None:
    train_path = cfg["dataset"]["train_file"]
    val_path = cfg["dataset"]["val_file"]
    missing = [p for p in [train_path, val_path] if not Path(p).exists()]
    if missing:
        print(
            f"[ERROR] Dataset files not found: {missing}\n"
            "Run `python data/prepare_sft_dataset.py` first."
        )
        sys.exit(1)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    check_dataset_exists(cfg)

    # --- Model & tokenizer ---
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.config)

    # --- Datasets ---
    print("Loading tokenized datasets...")
    train_dataset = SFTTraceDataset(
        cfg["dataset"]["train_file"],
        tokenizer,
        max_length=cfg["model"]["max_seq_length"],
    )
    val_dataset = SFTTraceDataset(
        cfg["dataset"]["val_file"],
        tokenizer,
        max_length=cfg["model"]["max_seq_length"],
    )
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # --- Training arguments ---
    tcfg = cfg["training"]
    output_dir = tcfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=tcfg["logging_steps"],
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        fp16=tcfg["fp16"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to=tcfg["report_to"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        eval_strategy="steps",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # --- Custom collator that handles our pre-masked labels ---
    pad_collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # --- Trainer ---
    # We use plain Trainer (not SFTTrainer) because our dataset already returns
    # tokenized tensors with instruction-masked labels. SFTTrainer would crash
    # when given a pre-tokenized dataset with no "text" field.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=pad_collate,
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save final adapter ---
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter → {final_dir}")


if __name__ == "__main__":
    main()
