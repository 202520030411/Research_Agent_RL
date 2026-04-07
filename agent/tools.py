"""
Mock search/read tools backed by HotpotQA context paragraphs.

In a real deployment these would call a live search API. Here we build an
in-memory index from the HotpotQA validation set so the agent can retrieve
relevant paragraphs without internet access during evaluation.

ToolExecutor.search(query) -> str   : returns top-k paragraph snippets
ToolExecutor.read(doc_text)  -> str : returns the full paragraph for a snippet
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap(a: str, b: str) -> float:
    """Simple token overlap score between two strings."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ToolExecutor:
    """
    Provides search and read tools backed by a static paragraph index.

    Uses an inverted token index for O(unique_query_tokens * postings) search
    instead of O(N) linear scan. On a 66k-paragraph index this is ~100x faster.

    Args:
        index_path : optional path to a pre-built JSONL index file
        top_k      : number of paragraphs returned per search
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        top_k: int = 2,
    ):
        self.top_k    = top_k
        self._index: list[dict] = []          # list of {title, text}
        self._full:  dict[str, str] = {}      # title -> full text
        self._inv:   dict[str, list[int]] = {}  # token -> [paragraph ids]

        if index_path and Path(index_path).exists():
            self._load_index(index_path)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_from_hotpotqa(self, hf_dataset, index_path: Optional[str] = None) -> None:
        """
        Populate the index from a HuggingFace HotpotQA dataset split.

        Args:
            hf_dataset : HuggingFace dataset object (e.g. hf["validation"])
            index_path : if given, save the index to this JSONL file for reuse
        """
        seen: set[str] = set()
        for example in hf_dataset:
            for title, sentences in zip(
                example["context"]["title"],
                example["context"]["sentences"],
            ):
                if title in seen:
                    continue
                seen.add(title)
                full_text = " ".join(sentences)
                entry = {"title": title, "text": full_text}
                self._index.append(entry)
                self._full[title] = full_text

        self._rebuild_inverted()

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                for entry in self._index:
                    f.write(json.dumps(entry) + "\n")
            print(f"Tool index built: {len(self._index)} paragraphs → {index_path}")

    def build_from_traces(self, jsonl_path: str, index_path: Optional[str] = None) -> None:
        """
        Build a lightweight index from the existing SFT trace JSONL file by
        extracting the document snippets embedded in read steps. Useful when
        the full HotpotQA dataset is not available.
        """
        seen: set[str] = set()
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                for step in rec.get("trace", []):
                    if step.get("action") == "read":
                        doc = step.get("document", "")
                        key = doc[:60]
                        if key not in seen:
                            seen.add(key)
                            entry = {"title": key, "text": doc}
                            self._index.append(entry)
                            self._full[key] = doc

        self._rebuild_inverted()

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                for entry in self._index:
                    f.write(json.dumps(entry) + "\n")
        print(f"Tool index built from traces: {len(self._index)} snippets")

    def _add_to_inverted(self, entry: dict, idx: int) -> None:
        tokens = set(_normalize(entry["title"] + " " + entry["text"]).split())
        for tok in tokens:
            self._inv.setdefault(tok, []).append(idx)

    def _rebuild_inverted(self) -> None:
        self._inv = {}
        for idx, entry in enumerate(self._index):
            self._add_to_inverted(entry, idx)

    def _load_index(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self._index.append(entry)
                self._full[entry["title"]] = entry["text"]
        self._rebuild_inverted()
        print(f"Tool index loaded: {len(self._index)} paragraphs from {path}")

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """
        Return the top-k most relevant paragraph snippets for the query.
        Uses inverted index to avoid O(N) linear scan.
        """
        if not self._index:
            return "[No index loaded. Call build_from_hotpotqa() first.]"

        query_tokens = set(_normalize(query).split())
        if not query_tokens:
            return f"[Empty query]"

        # Collect candidate paragraphs that share at least one token
        candidate_ids: dict[int, int] = {}  # idx -> shared token count
        for tok in query_tokens:
            for idx in self._inv.get(tok, []):
                candidate_ids[idx] = candidate_ids.get(idx, 0) + 1

        if not candidate_ids:
            return f"[No results found for: {query}]"

        # Score candidates by Jaccard overlap (only over candidates, not all N)
        scored = []
        for idx, hits in candidate_ids.items():
            entry = self._index[idx]
            score = _token_overlap(query, entry["title"] + " " + entry["text"])
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]

        if not top or top[0][0] == 0.0:
            return f"[No results found for: {query}]"

        parts = []
        for score, entry in top:
            snippet = entry["text"][:250].rstrip() + "..."
            parts.append(f"[{entry['title']}] {snippet}")
        return "\n\n".join(parts)

    def read(self, doc_text: str) -> str:
        """
        Given a document snippet (as returned by search or from a prior step),
        return the full paragraph it belongs to.
        """
        if not self._index:
            return "[No index loaded.]"

        # Try exact match by title key first
        if doc_text in self._full:
            return self._full[doc_text]

        # Otherwise find the paragraph with highest overlap to the snippet
        scored = [
            (_token_overlap(doc_text, e["text"]), e)
            for e in self._index
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0.1:
            return scored[0][1]["text"]

        return f"[Could not find full document for snippet: {doc_text[:80]}...]"
