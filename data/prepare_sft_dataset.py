"""
Prepare SFT dataset from HotpotQA.

Each HotpotQA example is turned into a multi-step trace that uses a REAL
ToolExecutor call, so the model trains on the same Observation: format it
sees at inference.

Retrieval strategy (fast version):
  For each HotpotQA example we build a tiny `ToolExecutor` over that
  example's own 10 candidate paragraphs (2 gold + 8 distractor) and run
  `search()` on it. That's 10 docs, not 90,000, so BM25 is O(tens of us).
  The val-split *global* index is still built + saved to disk so Weeks 2/3
  can reuse it at inference.

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

`document` in the read step is a short reference (doc_id) -- full text is in
`observation`.

Traces are flushed to JSONL every FLUSH_EVERY records so a timeout doesn't
lose all the work.
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

FLUSH_EVERY = 500  # write partial JSONL every N kept records


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


def build_mini_tool(example: dict) -> ToolExecutor:
    """Tiny ToolExecutor over this example's ~10 candidate paragraphs."""
    tool = ToolExecutor(top_k=2)
    tool._index = []
    tool._full = {}
    tool._title_to_id = {}
    seen_titles = set()
    for idx, (title, sentences) in enumerate(
        zip(example["context"]["title"], example["context"]["sentences"]),
        start=1,
    ):
        norm = _normalize_title(title)
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        text_parts = []
        seen_sents = set()
        for s in sentences:
            clean = re.sub(r"\s+", " ", s).strip()
            if clean and clean not in seen_sents:
                text_parts.append(clean)
                seen_sents.add(clean)
        text = " ".join(text_parts)
        doc_id = f"doc_{idx:05d}"
        tool._index.append({"doc_id": doc_id, "title": title, "text": text})
        tool._full[doc_id] = text
        tool._full[title] = text
        tool._title_to_id[norm] = doc_id
    tool._rebuild_inverted()
    return tool


def _search_hit_contains(search_obs: str, doc_id: str) -> bool:
    return f"[{doc_id}]" in search_obs


def build_trace(example: dict, cfg: dict) -> list[dict] | None:
    """
    Build a multi-step trace using real tool output on a per-example index.
    Returns None if BM25 fails to surface a supporting doc in its top-k.
    """
    question = example["question"]
    answer = example["answer"]
    supporting_titles = extract_supporting_titles(example)
    max_steps = cfg["dataset"]["max_steps_per_trace"]
    entities_to_search = supporting_titles[: max(1, max_steps // 2)]
    if not entities_to_search:
        return None

    tool = build_mini_tool(example)

    steps_raw = []
    for entity in entities_to_search:
        query = build_search_query(question, entity)
        search_obs = tool.search(query)
        doc_id = tool._title_to_id.get(_normalize_title(entity))
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
    output_path: str,
    desc: str,
) -> list[dict]:
    """
    Build traces for one split, flushing every FLUSH_EVERY kept records so
    a timeout/kernel-kill doesn't lose everything.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    last_flushed = 0
    dropped_no_hit = 0
    dropped_err = 0
    indices = list(range(len(hf_dataset)))
    random.shuffle(indices)

    pbar = tqdm(indices, desc=desc)
    for idx in pbar:
        if len(records) >= n_target:
            break
        example = hf_dataset[idx]
        try:
            trace = build_trace(example, cfg)
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
        if len(records) - last_flushed >= FLUSH_EVERY:
            _write_jsonl(records, output_path)
            last_flushed = len(records)
        pbar.set_postfix(kept=len(records), drop_nohit=dropped_no_hit, drop_err=dropped_err)

    _write_jsonl(records, output_path)  # final flush
    print(
        f"{desc}: kept={len(records)}  "
        f"dropped(no BM25 hit)={dropped_no_hit}  dropped(err)={dropped_err}  "
        f"-> {output_path}"
    )
    return records


def _write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    random.seed(42)
    cfg = load_config()

    print("Loading HotpotQA (distractor split)...")
    hf = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train_ds = hf["train"]
    val_ds = hf["validation"]

    train_size = cfg["dataset"]["train_size"]
    val_size = cfg["dataset"]["val_size"]

    # Build + save the global val-split index so Weeks 2/3 reuse it.
    # This is a one-time cost (~1 min for ~7k docs) and does NOT participate
    # in trace generation -- traces use per-example mini-indices below.
    val_tool_index = cfg["dataset"].get("tool_index_path", "data/sft_traces/tool_index.jsonl")
    if not Path(val_tool_index).exists():
        print("Building + saving val-split global ToolExecutor index...")
        val_tool = ToolExecutor(top_k=2)
        val_tool.build_from_hotpotqa(val_ds, index_path=val_tool_index)
    else:
        print(f"val-split tool index already exists -> {val_tool_index}")

    # Generate traces using per-example mini-indices (fast).
    process_split(
        train_ds, cfg, train_size,
        cfg["dataset"]["train_file"], "Building train traces",
    )
    process_split(
        val_ds, cfg, val_size,
        cfg["dataset"]["val_file"], "Building val traces",
    )

    # Quick sample print from the final train file.
    train_path = cfg["dataset"]["train_file"]
    if Path(train_path).exists():
        with open(train_path) as f:
            sample = json.loads(f.readline())
        print("\n--- Sample trace ---")
        print("Q:", sample["question"])
        print("A:", sample["answer"])
        for step in sample["trace"]:
            print(json.dumps({
                k: (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
                for k, v in step.items()
            }))
        print("--------------------")


if __name__ == "__main__":
    main()
