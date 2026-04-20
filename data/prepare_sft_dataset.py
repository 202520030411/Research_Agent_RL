"""
Prepare SFT dataset from HotpotQA.

Each HotpotQA example is turned into a multi-step trace that uses the REAL
`ToolExecutor.search()` and `ToolExecutor.read()` so the model trains on the
same observations it will see at inference. Examples where BM25 does not
retrieve the supporting document are dropped.

Output JSONL format per line:
  {
    "question": str,
    "answer": str,
    "trace": [
      {"thought": str, "action": "search", "query": str, "confidence": float,
       "observation": str},
      {"thought": str, "action": "read",   "document": str, "confidence": float,
       "observation": str},
      {"thought": str, "action": "answer", "answer": str, "confidence": float}
    ]
  }

`document` in the read step is a short reference (doc_id or title) — the full
text lives in `observation`.
"""

import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import ToolExecutor


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_supporting_titles(example: dict) -> list[str]:
    return list(dict.fromkeys(example["supporting_facts"]["title"]))


def build_search_query(question: str, entity_hint: str) -> str:
    """Entity + up to 3 content words from the question."""
    q_clean = question.strip().rstrip("?")
    entity_tokens = set(entity_hint.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                  "to", "and", "or", "did", "do", "does", "both", "same",
                  "which", "what", "who", "when", "where", "how", "that",
                  "have", "has", "had", "with", "for", "from", "by"}
    key_words = [
        w.strip("?,.")
        for w in q_clean.split()
        if w.lower().strip("?,.") not in stop_words
        and w.lower().strip("?,.") not in entity_tokens
        and len(w.strip("?,.")) > 3
    ][:3]
    if key_words:
        return f"{entity_hint} {' '.join(key_words)}"
    return entity_hint


def assign_confidence(step_idx: int, total_steps: int, cfg: dict) -> float:
    lo = cfg["dataset"]["confidence_schedule"]["min"]
    hi = cfg["dataset"]["confidence_schedule"]["max"]
    if total_steps == 1:
        return hi
    base = lo + (hi - lo) * (step_idx / (total_steps - 1))
    jitter = random.uniform(-0.04, 0.04)
    return round(min(max(base + jitter, lo), hi), 2)


def make_search_thought(question: str, entity: str) -> str:
    q_short = question.strip().rstrip("?")
    templates = [
        f"The question asks about \"{q_short}\". I need to look up \"{entity}\" to find relevant facts.",
        f"To answer \"{q_short}\", I should search for \"{entity}\" first.",
        f"Let me search for \"{entity}\" to gather information relevant to the question.",
        f"I need information about \"{entity}\" to answer the question about {q_short}.",
    ]
    return random.choice(templates)


def make_read_thought(entity: str) -> str:
    templates = [
        f"The search returned a document about \"{entity}\". Let me read it in full.",
        f"I found a result for \"{entity}\". I should read the full document now.",
        f"Reading the document about \"{entity}\" to get the details.",
    ]
    return random.choice(templates)


def make_answer_thought(question: str, entities: list[str]) -> str:
    entity_str = " and ".join(f'"{e}"' for e in entities) if entities else "the retrieved documents"
    q_short = question.strip().rstrip("?")
    templates = [
        f"Based on information gathered about {entity_str}, I can now answer \"{q_short}\".",
        f"After reading documents about {entity_str}, the answer to \"{q_short}\" is clear.",
        f"The documents about {entity_str} together answer \"{q_short}\".",
    ]
    return random.choice(templates)


def _doc_id_for_title(tool: ToolExecutor, title: str) -> str | None:
    return tool._title_to_id.get(_normalize_title(title))


def _search_hit_contains(search_obs: str, doc_id: str) -> bool:
    """True if the search output includes the given doc_id in its top-k."""
    return f"[{doc_id}]" in search_obs


def build_trace(example: dict, cfg: dict, tool: ToolExecutor) -> list[dict] | None:
    """
    Build a multi-step trace using real tool output.

    Returns None if BM25 fails to retrieve a supporting doc for any entity —
    the caller drops such examples.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_titles = extract_supporting_titles(example)
    max_steps = cfg["dataset"]["max_steps_per_trace"]
    entities_to_search = supporting_titles[: max(1, max_steps // 2)]

    if not entities_to_search:
        return None

    steps_raw = []
    for entity in entities_to_search:
        query = build_search_query(question, entity)
        search_obs = tool.search(query)

        doc_id = _doc_id_for_title(tool, entity)
        if doc_id is None or not _search_hit_contains(search_obs, doc_id):
            return None

        read_obs = tool.read(doc_id)

        steps_raw.append(("search", {"query": query, "_entity": entity, "observation": search_obs}))
        steps_raw.append(("read",   {"document": doc_id, "_entity": entity, "observation": read_obs}))

    steps_raw.append(("answer", {"answer": answer}))
    total = len(steps_raw)

    trace = []
    for i, (action, payload) in enumerate(steps_raw):
        conf = assign_confidence(i, total, cfg)
        if action == "search":
            trace.append({
                "thought": make_search_thought(question, payload["_entity"]),
                "action": "search",
                "query": payload["query"],
                "confidence": conf,
                "observation": payload["observation"],
            })
        elif action == "read":
            trace.append({
                "thought": make_read_thought(payload["_entity"]),
                "action": "read",
                "document": payload["document"],
                "confidence": conf,
                "observation": payload["observation"],
            })
        else:
            trace.append({
                "thought": make_answer_thought(question, entities_to_search),
                "action": "answer",
                "answer": answer,
                "confidence": conf,
            })
    return trace


def process_split(
    hf_dataset,
    cfg: dict,
    n_target: int,
    tool: ToolExecutor,
    desc: str,
) -> list[dict]:
    records = []
    dropped_no_hit = 0
    dropped_err = 0
    indices = list(range(len(hf_dataset)))
    random.shuffle(indices)

    pbar = tqdm(indices, desc=desc, total=min(n_target, len(indices)))
    for idx in pbar:
        if len(records) >= n_target:
            break
        example = hf_dataset[idx]
        try:
            trace = build_trace(example, cfg, tool)
        except Exception:
            dropped_err += 1
            continue
        if trace is None:
            dropped_no_hit += 1
            continue
        records.append({
            "question": example["question"],
            "answer": example["answer"],
            "trace": trace,
        })
        pbar.set_postfix(kept=len(records), drop_nohit=dropped_no_hit, drop_err=dropped_err)

    print(
        f"{desc}: kept={len(records)}  "
        f"dropped(no BM25 hit)={dropped_no_hit}  dropped(err)={dropped_err}"
    )
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records -> {path}")


def main():
    random.seed(42)
    cfg = load_config()

    print("Loading HotpotQA (distractor split)...")
    hf = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train_ds = hf["train"]
    val_ds = hf["validation"]

    train_size = cfg["dataset"]["train_size"]
    val_size = cfg["dataset"]["val_size"]

    # Train and val questions reference their own split's documents, so we
    # build one retrieval index per split. The val-split index is saved to
    # disk and reused at eval time (Weeks 2/3) via ToolExecutor(index_path=...).
    print("Building ToolExecutor index from HotpotQA train split...")
    train_tool = ToolExecutor(top_k=2)
    train_tool.build_from_hotpotqa(train_ds)
    print(f"  train corpus: {len(train_tool)} unique documents")

    val_tool_index = cfg["dataset"].get("tool_index_path", "data/sft_traces/tool_index.jsonl")
    print("Building ToolExecutor index from HotpotQA validation split...")
    val_tool = ToolExecutor(top_k=2)
    val_tool.build_from_hotpotqa(val_ds, index_path=val_tool_index)
    print(f"  val corpus: {len(val_tool)} unique documents")

    train_records = process_split(train_ds, cfg, train_size, train_tool, "Building train traces")
    val_records = process_split(val_ds, cfg, val_size, val_tool, "Building val traces")

    write_jsonl(train_records, cfg["dataset"]["train_file"])
    write_jsonl(val_records, cfg["dataset"]["val_file"])

    if train_records:
        sample = train_records[0]
        print("\n--- Sample trace ---")
        print("Q:", sample["question"])
        print("A:", sample["answer"])
        for step in sample["trace"]:
            print(json.dumps({k: (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
                              for k, v in step.items()}))
        print("--------------------")


if __name__ == "__main__":
    main()
