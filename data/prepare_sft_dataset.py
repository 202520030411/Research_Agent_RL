"""
Prepare SFT dataset from HotpotQA.

Converts HotpotQA examples into multi-step reasoning traces in structured JSON
format. Each trace simulates the search-read-answer loop the agent must learn.

Output JSONL format per line:
  {
    "question": str,
    "trace": [
      {"thought": str, "action": "search", "query": str, "confidence": float},
      {"thought": str, "action": "read",   "document": str, "confidence": float},
      ...
      {"thought": str, "action": "answer", "answer": str, "confidence": float}
    ]
  }
"""

import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_supporting_titles(example: dict) -> list[str]:
    """Return the titles of paragraphs marked as supporting facts."""
    supporting = {fact[0] for fact in example["supporting_facts"]["title"]}
    return list(supporting)


def build_search_query(question: str, entity_hint: str) -> str:
    """Heuristically build a search query for an entity given the question."""
    q = question.strip().rstrip("?")
    # If entity is already prominent in the question, use it directly
    if entity_hint.lower() in q.lower():
        return entity_hint
    return f"{entity_hint}"


def assign_confidence(step_idx: int, total_steps: int, cfg: dict) -> float:
    """
    Assign confidence score that increases as the agent gathers more evidence.
    Early steps have low confidence; the final answer step has high confidence.
    """
    lo = cfg["dataset"]["confidence_schedule"]["min"]
    hi = cfg["dataset"]["confidence_schedule"]["max"]
    if total_steps == 1:
        return hi
    # Linear ramp from lo to hi, with small random jitter for realism
    base = lo + (hi - lo) * (step_idx / (total_steps - 1))
    jitter = random.uniform(-0.04, 0.04)
    return round(min(max(base + jitter, lo), hi), 2)


def make_search_thought(question: str, entity: str, step_idx: int) -> str:
    templates = [
        f"I need to find information about {entity} to answer the question.",
        f"To answer this, I should look up {entity} first.",
        f"Let me search for {entity} to gather relevant facts.",
        f"The question asks about {entity}. I will search for it.",
    ]
    return random.choice(templates)


def make_read_thought(entity: str, doc_snippet: str) -> str:
    snippet = doc_snippet[:120].rstrip() + ("..." if len(doc_snippet) > 120 else "")
    templates = [
        f"I found a document about {entity}. It states: \"{snippet}\"",
        f"The document on {entity} provides: \"{snippet}\"",
        f"Reading about {entity}: \"{snippet}\"",
    ]
    return random.choice(templates)


def make_answer_thought(question: str, answer: str, entities: list[str]) -> str:
    entity_str = " and ".join(entities) if entities else "the retrieved documents"
    templates = [
        f"Based on the information gathered about {entity_str}, the answer to the question is clear.",
        f"After reviewing documents on {entity_str}, I can now answer confidently.",
        f"The documents about {entity_str} together provide the answer.",
    ]
    return random.choice(templates)


def get_doc_snippet(example: dict, title: str) -> str:
    """Retrieve the first sentence(s) of a paragraph by title."""
    for t, sentences in zip(
        example["context"]["title"], example["context"]["sentences"]
    ):
        if t == title:
            combined = " ".join(sentences)
            # Return first ~200 chars
            return combined[:200]
    return ""


def build_trace(example: dict, cfg: dict) -> list[dict]:
    """
    Build a multi-step reasoning trace for one HotpotQA example.

    Strategy:
      - For each supporting entity (up to max_steps//2), emit a search + read step.
      - End with an answer step.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_titles = extract_supporting_titles(example)
    max_steps = cfg["dataset"]["max_steps_per_trace"]

    # Cap at floor(max_steps/2) search-read pairs + 1 answer = max_steps+1 steps max
    entities_to_search = supporting_titles[: max(1, max_steps // 2)]

    steps = []
    for entity in entities_to_search:
        query = build_search_query(question, entity)
        doc_snippet = get_doc_snippet(example, entity)

        steps.append({
            "action": "search",
            "query": query,
            "_thought_entity": entity,
        })
        if doc_snippet:
            steps.append({
                "action": "read",
                "document": doc_snippet,
                "_thought_entity": entity,
            })

    steps.append({"action": "answer", "answer": answer})

    total = len(steps)
    trace = []
    for i, step in enumerate(steps):
        conf = assign_confidence(i, total, cfg)
        action = step["action"]

        if action == "search":
            thought = make_search_thought(question, step["_thought_entity"], i)
            trace.append({
                "thought": thought,
                "action": "search",
                "query": step["query"],
                "confidence": conf,
            })
        elif action == "read":
            thought = make_read_thought(step["_thought_entity"], step["document"])
            trace.append({
                "thought": thought,
                "action": "read",
                "document": step["document"],
                "confidence": conf,
            })
        else:  # answer
            thought = make_answer_thought(question, answer, entities_to_search)
            trace.append({
                "thought": thought,
                "action": "answer",
                "answer": answer,
                "confidence": conf,
            })

    return trace


def process_split(
    hf_dataset,
    cfg: dict,
    n_samples: int,
    desc: str,
) -> list[dict]:
    records = []
    indices = list(range(len(hf_dataset)))
    random.shuffle(indices)

    for idx in tqdm(indices[:n_samples], desc=desc):
        example = hf_dataset[idx]
        try:
            trace = build_trace(example, cfg)
        except Exception:
            continue
        records.append({
            "question": example["question"],
            "answer": example["answer"],
            "trace": trace,
        })

    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records → {path}")


def main():
    random.seed(42)
    cfg = load_config()

    print("Loading HotpotQA (distractor split)...")
    hf = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train_ds = hf["train"]
    val_ds = hf["validation"]

    train_size = cfg["dataset"]["train_size"]
    val_size = cfg["dataset"]["val_size"]

    train_records = process_split(train_ds, cfg, train_size, "Building train traces")
    val_records = process_split(val_ds, cfg, val_size, "Building val traces")

    write_jsonl(train_records, cfg["dataset"]["train_file"])
    write_jsonl(val_records, cfg["dataset"]["val_file"])

    # Quick sanity print
    sample = train_records[0]
    print("\n--- Sample trace ---")
    print("Q:", sample["question"])
    for step in sample["trace"]:
        print(json.dumps(step))
    print("--------------------")


if __name__ == "__main__":
    main()
