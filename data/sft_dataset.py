"""
SFT Dataset: tokenizes reasoning traces into model input format.

Format matches agent inference exactly:

  <|im_start|>system
  You are a research agent ...
  <|im_end|>
  <|im_start|>user
  Question: {question}
  <|im_end|>
  <|im_start|>assistant
  Step 1: {"thought": ..., "action": "search", "query": ..., "confidence": ...}
  Observation: [doc_00042] Title :: snippet...
  Step 2: {"thought": ..., "action": "read", "document": "doc_00042", ...}
  Observation: <full document text>
  Step 3: {"thought": ..., "action": "answer", "answer": ..., "confidence": ...}
  <|im_end|>

Loss masking:
  - System + user tokens -> IGNORE_INDEX
  - Observation: ... lines -> IGNORE_INDEX (environment output, not model output)
  - Step N: {...} lines -> kept (this is what the model learns to generate)
"""

import json
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100
OBS_PREVIEW = 300  # chars of observation exposed in the trace

SYSTEM_PROMPT = (
    "You are a research agent. Given a question, reason step by step and use "
    "tools (search, read) to gather information. At each step output exactly one "
    "JSON object with keys: thought, action, and action-specific fields "
    "(query for search, document for read, answer for answer), plus a confidence "
    "score between 0 and 1. Output one JSON object per line."
)


def _strip_observation(step: dict) -> dict:
    return {k: v for k, v in step.items() if k != "observation"}


def format_trace_as_text(trace: list[dict]) -> str:
    """
    Render a trace as assistant content with Observation: lines after
    search/read steps. The observation field is pulled out of each step dict
    and rendered as a separate line -- it is not part of the JSON the model
    generates.
    """
    lines = []
    for i, step in enumerate(trace, start=1):
        step_json = _strip_observation(step)
        lines.append(f"Step {i}: {json.dumps(step_json)}")
        obs = step.get("observation")
        if obs:
            lines.append(f"Observation: {obs[:OBS_PREVIEW]}")
    return "\n".join(lines)


def build_chat_string(
    question: str,
    trace: list[dict],
    tokenizer: PreTrainedTokenizer,
) -> tuple[str, str]:
    """
    Return (full_chat_string, assistant_content_string).

    The second value is the exact substring of the first that contains the
    assistant turn's content -- used to compute observation char spans.
    """
    assistant_content = format_trace_as_text(trace)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": assistant_content},
    ]
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return full, assistant_content


def compute_observation_char_spans(
    assistant_content: str,
    asst_start_in_full: int,
) -> list[tuple[int, int]]:
    """
    Return char spans [start, end) in the FULL chat string that fall inside
    Observation: lines (including the "Observation: " prefix itself) -- these
    tokens will be masked from the loss.

    `asst_start_in_full` is the char index where assistant content begins
    inside the full chat string.
    """
    spans = []
    offset = 0
    for line in assistant_content.split("\n"):
        if line.startswith("Observation:"):
            start = asst_start_in_full + offset
            end = start + len(line)
            spans.append((start, end))
        offset += len(line) + 1  # +1 for the '\n' separator
    return spans


class SFTTraceDataset(Dataset):
    """
    Loads a JSONL file of reasoning traces and tokenizes them on-the-fly.

    Each item returns:
      input_ids      : (seq_len,) LongTensor
      labels         : (seq_len,) LongTensor -- IGNORE_INDEX on prompt + observations
      attention_mask : (seq_len,) LongTensor
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

        full_text, assistant_content = build_chat_string(
            question, trace, self.tokenizer
        )

        # Locate assistant content's char offset inside the full chat string.
        asst_start_char = full_text.find(assistant_content)
        if asst_start_char < 0:
            # Chat template mangled the content somehow; fall back to training
            # on everything (rare; prior behavior).
            asst_start_char = 0
        asst_end_char = asst_start_char + len(assistant_content)

        obs_spans = compute_observation_char_spans(assistant_content, asst_start_char)

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]

        labels = [IGNORE_INDEX] * len(input_ids)
        for i, (c0, c1) in enumerate(offsets):
            if c0 == c1:
                continue  # special tokens with empty spans
            # Keep tokens strictly inside the assistant content...
            if c0 < asst_start_char or c1 > asst_end_char:
                continue
            # ...but drop any that fall inside an Observation: line.
            in_obs = any(c0 < end and c1 > start for start, end in obs_spans)
            if in_obs:
                continue
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
