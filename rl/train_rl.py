"""
REINFORCE training loop for the RL stopping policy.

Algorithm:
  1. Collect N episodes using the current policy (rollout.py).
  2. For each episode, the reward is a single scalar R (terminal reward).
  3. Baseline: subtract the running mean reward to reduce variance.
  4. Policy gradient loss: -Σ_t log π(a_t | s_t) * (R - baseline)
  5. Update policy network with Adam.
  6. Repeat for K epochs.

The SFT LLM is frozen throughout — only the tiny MLP is updated.

Usage (from project root):
    python -m rl.train_rl --config config.yaml --checkpoint checkpoints/qwen-sft-adapter
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset

from sft.model import load_model_and_tokenizer
from agent.tools import ToolExecutor
from rl.policy_network import PolicyNetwork
from rl.rollout import RolloutCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_hotpotqa_questions(split: str, n: int) -> list[dict]:
    """Load n questions from HotpotQA (distractor split)."""
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=42)
    return [
        {"question": row["question"], "answer": row["answer"], "contexts": row["context"]}
        for row in ds.select(range(min(n, len(ds))))
    ]


def build_baseline(rewards: list[float], alpha: float = 0.05) -> float:
    """Exponential moving average baseline."""
    if not rewards:
        return 0.0
    baseline = rewards[0]
    for r in rewards[1:]:
        baseline = (1 - alpha) * baseline + alpha * r
    return baseline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config_path:    str,
    checkpoint_dir: str,
    output_dir:     str,
    n_train_q:      int = 1000,
    n_val_q:        int = 100,
    n_episodes:     int = 16,    # episodes per policy update
    n_epochs:       int = 50,
    lr:             float = 3e-4,
    max_steps:      int = 6,
    device:         str = "cuda",
    save_every:     int = 10,
    seed:           int = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = load_config(config_path)

    rcfg     = cfg["reward"]
    alpha    = rcfg["alpha"]
    beta     = rcfg["beta"]
    epsilon  = rcfg["epsilon"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading SFT model (frozen)...")
    model, tokenizer = load_model_and_tokenizer(config_path)

    # Load LoRA checkpoint if provided
    if checkpoint_dir:
        from peft import PeftModel
        print(f"  Loading LoRA adapters from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    print("Building tool executor...")
    train_questions = load_hotpotqa_questions("train", n_train_q)
    val_questions   = load_hotpotqa_questions("validation", n_val_q)

    tool_executor = ToolExecutor(top_k=2)
    tool_executor.build_from_hotpotqa(
        load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True),
        index_path="data/tool_index.jsonl",
    )

    print("Initialising RL policy network...")
    policy = PolicyNetwork(hidden=64).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    collector = RolloutCollector(
        sft_model=model,
        tokenizer=tokenizer,
        tool_executor=tool_executor,
        policy_network=policy,
        max_steps=max_steps,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
    )

    # Running reward baseline (EMA)
    running_rewards: list[float] = []
    history_log: list[dict] = []

    print(f"\nStarting REINFORCE training for {n_epochs} epochs, "
          f"{n_episodes} episodes/epoch...")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        policy.train()

        # --- Collect episodes ---
        batch_qs = random.sample(train_questions, min(n_episodes, len(train_questions)))
        episodes = collector.collect_batch(batch_qs, n_episodes=n_episodes)

        raw_rewards = [ep.reward.total for ep in episodes]
        running_rewards.extend(raw_rewards)
        baseline = sum(running_rewards[-50:]) / len(running_rewards[-50:])

        # --- REINFORCE loss ---
        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_transitions = 0

        for ep in episodes:
            R = torch.tensor(ep.reward.total, device=device)
            advantage = R - baseline

            for trans in ep.transitions:
                log_prob      = trans.log_prob.to(device)
                policy_loss   = policy_loss - log_prob * advantage
                n_transitions += 1

        if n_transitions > 0:
            policy_loss = policy_loss / n_transitions
            optimiser.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimiser.step()

        # --- Logging ---
        acc         = sum(1 for ep in episodes if ep.result.correct) / len(episodes)
        avg_steps   = sum(ep.result.n_steps for ep in episodes) / len(episodes)
        avg_reward  = sum(raw_rewards) / len(raw_rewards)
        elapsed     = time.time() - t0

        log_entry = {
            "epoch":      epoch,
            "loss":       policy_loss.item(),
            "reward":     avg_reward,
            "baseline":   baseline,
            "accuracy":   acc,
            "avg_steps":  avg_steps,
            "elapsed_s":  round(elapsed, 1),
        }
        history_log.append(log_entry)
        print(
            f"Epoch {epoch:3d}/{n_epochs}  "
            f"loss={log_entry['loss']:+.4f}  "
            f"R={avg_reward:+.3f}  base={baseline:+.3f}  "
            f"acc={acc:.2f}  steps={avg_steps:.1f}  "
            f"({elapsed:.0f}s)"
        )

        # --- Validation ---
        if epoch % save_every == 0:
            policy.eval()
            val_eps = collector.collect_batch(
                random.sample(val_questions, min(50, len(val_questions)))
            )
            val_acc    = sum(1 for e in val_eps if e.result.correct) / len(val_eps)
            val_steps  = sum(e.result.n_steps for e in val_eps) / len(val_eps)
            val_reward = sum(e.reward.total for e in val_eps) / len(val_eps)
            print(
                f"  [VAL] acc={val_acc:.3f}  steps={val_steps:.1f}  "
                f"reward={val_reward:+.3f}"
            )
            ckpt_path = f"{output_dir}/policy_epoch{epoch:03d}.pt"
            policy.save(ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # --- Final save ---
    final_path = f"{output_dir}/policy_final.pt"
    policy.save(final_path)
    log_path = f"{output_dir}/training_log.json"
    with open(log_path, "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nTraining complete. Policy saved to {final_path}")
    print(f"Training log saved to {log_path}")
    return policy


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/qwen-sft-adapter",
                        help="Path to SFT LoRA adapter directory")
    parser.add_argument("--output",     default="checkpoints/rl-policy")
    parser.add_argument("--n_train",    type=int, default=1000)
    parser.add_argument("--n_val",      type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=16,
                        help="Episodes per policy update")
    parser.add_argument("--n_epochs",   type=int, default=50)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--max_steps",  type=int, default=6)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    train(
        config_path=args.config,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        n_train_q=args.n_train,
        n_val_q=args.n_val,
        n_episodes=args.n_episodes,
        n_epochs=args.n_epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        save_every=args.save_every,
        seed=args.seed,
    )
