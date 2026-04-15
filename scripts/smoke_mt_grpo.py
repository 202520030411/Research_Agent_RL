"""
Smoke test for batched multi-turn GRPO.

Verifies the new `collect_episodes_batched` pipeline end-to-end on CPU/MPS
with a tiny model and a stub ToolExecutor. Checks:

  1. G episodes are returned.
  2. Each Episode.full_text contains the base prompt + all turn texts.
  3. Turn char spans are valid and point inside full_text.
  4. compute_trajectory_log_probs runs without error and returns a finite scalar.
  5. One grpo_train_step completes end-to-end and produces a finite loss.

Run:  python scripts/smoke_mt_grpo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.multi_turn_grpo import (
    collect_episodes_batched,
    compute_trajectory_log_probs,
    grpo_train_step,
    SYSTEM_PROMPT_FALLBACK,
)


class StubToolExecutor:
    """Returns canned strings so we exercise the loop without HotpotQA."""

    def search(self, query: str) -> str:
        return f"[doc_00001] Fake Doc :: A document mentioning {query[:40]}."

    def read(self, doc_ref: str) -> str:
        return f"Full text of {doc_ref[:40]}: Alexander Graham Bell invented the telephone in 1876."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    print(f"device = {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tool = StubToolExecutor()

    questions = [
        {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
        {"question": "In what year was the telephone invented?", "answer": "1876"},
    ]

    G, max_steps = 2, 3

    # --- Test 1: batched rollout produces well-formed episodes -----------
    print("\n[1] collect_episodes_batched ...")
    episodes = collect_episodes_batched(
        question=questions[0]["question"],
        gold_answer=questions[0]["answer"],
        model=model,
        tokenizer=tokenizer,
        tool_executor=tool,
        system_prompt=SYSTEM_PROMPT_FALLBACK,
        G=G,
        max_steps=max_steps,
        max_new_tokens=80,
        temperature=0.9,
        device=device,
    )
    assert len(episodes) == G, f"expected {G} episodes, got {len(episodes)}"
    for i, ep in enumerate(episodes):
        assert ep.full_text, f"episode {i} has empty full_text"
        assert ep.turns,     f"episode {i} has no turns"
        for t_idx, rec in enumerate(ep.turns):
            assert 0 <= rec.char_start < rec.char_end <= len(ep.full_text), \
                f"ep{i} turn{t_idx} bad span {rec.char_start}..{rec.char_end} / {len(ep.full_text)}"
            slice_text = ep.full_text[rec.char_start:rec.char_end]
            assert slice_text == rec.text, \
                f"ep{i} turn{t_idx} span doesn't match recorded text"
        print(f"  ep{i}: n_turns={len(ep.turns)}  correct={ep.correct}  "
              f"len(full_text)={len(ep.full_text)}")

    # --- Test 2: trajectory log-prob is finite and has a gradient --------
    print("\n[2] compute_trajectory_log_probs ...")
    model.train()
    lp = compute_trajectory_log_probs(model, tokenizer, episodes[0], device=device)
    assert torch.isfinite(lp), f"log-prob not finite: {lp}"
    assert lp.requires_grad,    "log-prob tensor has no grad"
    print(f"  log_prob = {lp.item():.3f}  (requires_grad=True)")

    # --- Test 3: one full grpo_train_step end-to-end ---------------------
    print("\n[3] grpo_train_step ...")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-6
    )
    metrics = grpo_train_step(
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        tool_executor=tool,
        optimizer=optimizer,
        system_prompt=SYSTEM_PROMPT_FALLBACK,
        G=G,
        max_steps=max_steps,
        device=device,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"])), f"loss not finite: {metrics}"
    print(f"  loss={metrics['loss']:+.4f}  acc={metrics['accuracy']:.2f}  "
          f"avg_steps={metrics['avg_steps']:.1f}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
