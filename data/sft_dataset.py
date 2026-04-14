"""
SFT Dataset: tokenizes reasoning traces into model input format.

Each JSONL record is converted into a chat-formatted string:
  <|im_start|>system
  You are a research agent ...
  <|im_end|>
  <|im_start|>user
  Question: {question}
  <|im_end|>
  <|im_start|>assistant
  Step 1: {"thought": ..., "action": "search", ...}
  Step 2: {"thought": ..., "action": "read", ...}
  ...
  Step N: {"thought": ..., "action": "answer", ...}
  <|im_end|>

Loss is computed ONLY on the assistant turn tokens (instruction masking).
All system + user tokens are masked with IGNORE_INDEX = -100.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a research agent. Given a question, reason step by step and use "
    "tools (search, read) to gather information. At each step output exactly one "
    "JSON object with keys: thought, action, and action-specific fields "
    "(query for search, document for read, answer for answer), plus a confidence "
    "score between 0 and 1. Output one JSON object per line."
)


def format_trace_as_text(trace: list[dict]) -> str:
    """Convert a list of step dicts into the assistant response string."""
    lines = []
    for i, step in enumerate(trace, start=1):
        lines.append(f"Step {i}: {json.dumps(step)}")
    return "\n".join(lines)


def build_chat_string(question: str, trace: list[dict], tokenizer: PreTrainedTokenizer) -> str:
    """Build the full chat-formatted string using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": format_trace_as_text(trace)},
    ]
    # apply_chat_template with tokenize=False gives us the raw string
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def find_assistant_token_span(
    input_ids: list[int],
    tokenizer: PreTrainedTokenizer,
) -> tuple[int, int]:
    """
    Return (start, end) indices of the assistant content tokens so we can mask
    everything before them with IGNORE_INDEX.

    Returns (-1, -1) if the assistant marker is not found (e.g. truncation cut
    it off) — caller must skip the sample rather than train on prompt tokens.
    """
    assistant_start_marker = "<|im_start|>assistant"
    marker_ids = tokenizer.encode(assistant_start_marker, add_special_tokens=False)

    n = len(input_ids)
    m = len(marker_ids)
    for i in range(n - m, -1, -1):
        if input_ids[i : i + m] == marker_ids:
            return i + m, n

    return -1, -1


class SFTTraceDataset(Dataset):
    """
    Loads a JSONL file of reasoning traces and tokenizes them on-the-fly.

    Each item returns:
      input_ids  : (seq_len,) LongTensor
      labels     : (seq_len,) LongTensor  — IGNORE_INDEX everywhere except assistant turn
      attention_mask: (seq_len,) LongTensor
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = self._load(jsonl_path)

    def _load(self, path: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        question = record["question"]
        trace = record["trace"]

        full_text = build_chat_string(question, trace, self.tokenizer)

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Build labels: mask everything before the assistant content
        labels = [IGNORE_INDEX] * len(input_ids)
        asst_start, asst_end = find_assistant_token_span(input_ids, self.tokenizer)
        if asst_start == -1:
            # Assistant marker was truncated away — everything stays masked so
            # the sample contributes zero loss instead of training on the prompt.
            pass
        else:
            for i in range(asst_start, asst_end):
                labels[i] = input_ids[i]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Left-pad sequences in a batch to the same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, attn_list = [], [], []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                       item["input_ids"]])
        )
        labels_list.append(
            torch.cat([torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                       item["labels"]])
        )
        attn_list.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long),
                       item["attention_mask"]])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attn_list),
    }
