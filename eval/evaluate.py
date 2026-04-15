"""
Baseline evaluation script — Week 2.

Runs three stopping policies on a sample of HotpotQA validation questions
and prints a comparison table.

Usage (from repo root):
  python eval/evaluate.py
  python eval/evaluate.py --n_questions 100 --adapter checkpoints/qwen-sft/final
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import ResearchAgent
from agent.stopping import ConfidencePolicy, FixedStepPolicy, NeverStop
from agent.tools import ToolExecutor
from eval.metrics import compare_policies, compute_metrics, format_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 2 baseline evaluation")
    p.add_argument("--adapter",      default="checkpoints/qwen-sft/final",
                   help="Path to SFT LoRA adapter directory")
    p.add_argument("--n_questions",  type=int, default=100,
                   help="Number of HotpotQA val questions to evaluate")
    p.add_argument("--max_steps",    type=int, default=6,
                   help="Hard cap on steps per question")
    p.add_argument("--index_path",   default="data/tool_index.jsonl",
                   help="Path to pre-built tool index (built if missing)")
    p.add_argument("--val_traces",   default="data/sft_traces/val.jsonl",
                   help="Val JSONL used as fallback to build tool index")
    p.add_argument("--output",       default="eval/results.json",
                   help="Where to save per-question results")
    return p.parse_args()


def load_model(adapter_path: str):
    """Load the SFT model + tokenizer from the LoRA adapter checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model + adapter from {adapter_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Read base model name from adapter_config.json
    adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    model.config.use_cache = True  # re-enable for inference speed
    return model, tokenizer


def load_val_questions(n: int) -> list[dict]:
    """Load n questions from HotpotQA validation split."""
    from datasets import load_dataset
    print("Loading HotpotQA validation split...")
    hf = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    val = hf["validation"]
    return [{"question": val[i]["question"], "answer": val[i]["answer"],
             "context": val[i]["context"]}
            for i in range(min(n, len(val)))]


def build_tool_index(questions: list[dict], index_path: str,
                     fallback_traces: str) -> ToolExecutor:
    """Build the tool executor, using cache if available."""
    executor = ToolExecutor(index_path=index_path if Path(index_path).exists() else None)

    if len(executor) > 0:
        return executor

    # Build from HotpotQA context embedded in the questions
    if questions:
        executor.build_from_hotpotqa(questions, index_path=index_path)

    elif Path(fallback_traces).exists():
        executor.build_from_traces(fallback_traces, index_path)

    return executor


def run_policy(
    policy_name: str,
    policy,
    questions: list[dict],
    model,
    tokenizer,
    tool_executor: ToolExecutor,
    max_steps: int,
) -> list:
    from agent.agent import ResearchAgent
    agent = ResearchAgent(model, tokenizer, tool_executor, policy, max_steps=max_steps)
    results = []
    for q in tqdm(questions, desc=policy_name):
        policy.reset()
        result = agent.run(q["question"], gold_answer=q["answer"])
        results.append(result)
    return results


def main():
    args = parse_args()

    # --- Load model ---
    model, tokenizer = load_model(args.adapter)

    # --- Load questions ---
    questions = load_val_questions(args.n_questions)

    # --- Build tool index ---
    tool_executor = build_tool_index(questions, args.index_path, args.val_traces)

    # --- Define policies ---
    policies = {
        "FixedStep (N=2)":           FixedStepPolicy(max_steps=2),
        "FixedStep (N=3)":           FixedStepPolicy(max_steps=3),
        "Confidence (τ=0.85)":       ConfidencePolicy(threshold=0.85),
        "NeverStop (oracle steps)":  NeverStop(),
    }

    # --- Evaluate ---
    all_results: dict[str, list] = {}
    for name, policy in policies.items():
        print(f"\n{'='*55}\nEvaluating: {name}\n{'='*55}")
        results = run_policy(name, policy, questions, model, tokenizer,
                             tool_executor, args.max_steps)
        all_results[name] = results
        m = compute_metrics(results)
        print(format_metrics(m, name))

    # --- Print comparison table ---
    print(f"\n{'='*65}")
    print("COMPARISON TABLE")
    print(compare_policies(all_results))

    # --- Save results ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, results in all_results.items():
        serializable[name] = [
            {
                "question":     r.question,
                "gold_answer":  r.gold_answer,
                "pred_answer":  r.pred_answer,
                "correct":      r.correct,
                "n_steps":      r.n_steps,
                "n_tool_calls": r.n_tool_calls,
                "stopped_by":   r.stopped_by,
            }
            for r in results
        ]
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPer-question results saved → {args.output}")


if __name__ == "__main__":
    main()
