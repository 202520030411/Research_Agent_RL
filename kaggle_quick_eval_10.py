"""
Quick eval for the completed dense-reward GRPO checkpoint on Kaggle.

Use this in a fresh Kaggle notebook with:
  - GPU T4 enabled
  - Internet ON

This is eval-only and runs the normal 100-question final eval. It saves
partial results after every question.
"""

# %% 1. Install + clone repo
!pip install -q transformers peft accelerate datasets pyyaml tqdm bitsandbytes

import os
import json
import time

REPO_URL = "https://github.com/202520030411/Research_Agent_RL.git"
REPO_DIR = "/kaggle/working/Research_Agent_RL"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())


# %% 2. Download trained adapter from the completed Kaggle run
!mkdir -p /kaggle/working/week3grpo2
!kaggle kernels output wuyue22/week3grpo2 -p /kaggle/working/week3grpo2

candidates = [
    "/kaggle/working/week3grpo2/Research_Agent_RL/checkpoints/mt-grpo/final",
    "/kaggle/working/week3grpo2/Research_Agent_RL/checkpoints/mt-grpo/epoch200",
    # If you add the previous run as Kaggle input instead of downloading it:
    "/kaggle/input/week3grpo2/Research_Agent_RL/checkpoints/mt-grpo/final",
    "/kaggle/input/week3grpo2/Research_Agent_RL/checkpoints/mt-grpo/epoch200",
]

ADAPTER_DIR = None
for candidate in candidates:
    if os.path.exists(os.path.join(candidate, "adapter_config.json")):
        ADAPTER_DIR = candidate
        break

assert ADAPTER_DIR is not None, "Could not find final/epoch200 adapter."
print("Using adapter:", ADAPTER_DIR)
!ls -lh {ADAPTER_DIR}


# %% 3. Load model
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

with open(os.path.join(ADAPTER_DIR, "adapter_config.json")) as f:
    base_model_name = json.load(f).get(
        "base_model_name_or_path",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )

base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
model.eval()
model.config.use_cache = True

print("Loaded model on", DEVICE)


# %% 4. Quick eval; saves after every question
from datasets import load_dataset
from tqdm import tqdm

from agent.tools import ToolExecutor
from data.sft_dataset import SYSTEM_PROMPT
from rl.multi_turn_grpo import collect_episode

N_EVAL = 100
MAX_EVAL_SECONDS = 3 * 3600
OUT = "/kaggle/working/quick_eval_100.json"

hf = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
val_hf = hf["validation"]

val_questions = [
    {"question": val_hf[i]["question"], "answer": val_hf[i]["answer"]}
    for i in range(N_EVAL)
]

index_candidates = [
    "/kaggle/working/week3grpo2/Research_Agent_RL/data/tool_index.jsonl",
    "/kaggle/input/week3grpo2/Research_Agent_RL/data/tool_index.jsonl",
]
index_path = next((p for p in index_candidates if os.path.exists(p)), None)

tool_executor = ToolExecutor(index_path=index_path, top_k=2)
if len(tool_executor) == 0:
    tool_executor.build_from_hotpotqa(
        val_hf,
        index_path="/kaggle/working/tool_index.jsonl",
    )

results = []
correct = 0
total_steps = 0
t0 = time.time()

for idx, q in enumerate(tqdm(val_questions, desc="Quick eval")):
    if time.time() - t0 > MAX_EVAL_SECONDS:
        print("Stopping eval early to stay under time budget")
        break

    ep = collect_episode(
        q["question"],
        q["answer"],
        model,
        tokenizer,
        tool_executor,
        system_prompt=SYSTEM_PROMPT,
        max_steps=5,
        max_new_tokens=96,
        temperature=0.1,
        device=DEVICE,
    )

    correct += int(ep.correct)
    total_steps += ep.n_steps
    results.append({
        "idx": idx,
        "question": ep.question,
        "gold": ep.gold_answer,
        "correct": ep.correct,
        "n_steps": ep.n_steps,
        "tail": ep.full_text[-700:],
    })

    n = len(results)
    summary = {
        "n_eval_completed": n,
        "accuracy": correct / n,
        "avg_steps": total_steps / n,
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(OUT, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

print(summary)
print("Saved:", OUT)
