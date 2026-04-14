"""
Mock search/read tools backed by HotpotQA documents.

In a real deployment these would call a live search API. Here we build an
in-memory document store from the HotpotQA corpus so the agent can retrieve
ranked documents without internet access during training and evaluation.

ToolExecutor.search(query) -> str
    Returns top-k document hits as compact `doc_id` + title + snippet lines.

ToolExecutor.read(doc_ref) -> str
    Returns the full document text for a `doc_id`, title, or approximate
    snippet, which keeps older SFT traces backward-compatible.
"""

import json
import math
import re
import unicodedata
from collections import Counter
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


def _tokenize(text: str) -> list[str]:
    """Normalize then tokenize on whitespace."""
    return [tok for tok in _normalize(text).split() if tok]


class ToolExecutor:
    """
    Provides search and read tools backed by a static document index.

    Uses an inverted token index for O(unique_query_tokens * postings) search
    instead of O(N) linear scan.

    Args:
        index_path : optional path to a pre-built JSONL index file
        top_k      : number of documents returned per search
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        top_k: int = 2,
    ):
        self.top_k    = top_k
        self._index: list[dict] = []            # list of {doc_id, title, text}
        self._full:  dict[str, str] = {}        # doc_id/title -> full text
        self._title_to_id: dict[str, str] = {}  # normalized title -> doc_id
        self._inv:   dict[str, list[int]] = {}  # token -> [document ids]
        self._term_freqs: list[dict[str, int]] = []
        self._title_terms: list[set[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._doc_len: list[int] = []
        self._avg_doc_len: float = 0.0

        if index_path and Path(index_path).exists():
            self._load_index(index_path)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_from_hotpotqa(self, hf_dataset, index_path: Optional[str] = None) -> None:
        """
        Populate the index from a HuggingFace HotpotQA dataset split.

        Documents are aggregated by title across examples, then deduplicated
        sentence-by-sentence so `read()` can return a fuller article-like block
        instead of a single paragraph.

        Args:
            hf_dataset : HuggingFace dataset object (e.g. hf["validation"])
            index_path : if given, save the index to this JSONL file for reuse
        """
        docs: dict[str, dict] = {}
        for example in hf_dataset:
            for title, sentences in zip(
                example["context"]["title"],
                example["context"]["sentences"],
            ):
                norm_title = _normalize(title)
                if norm_title not in docs:
                    docs[norm_title] = {
                        "title": title,
                        "sentences": [],
                        "seen_sentences": set(),
                    }

                for sent in sentences:
                    clean = re.sub(r"\s+", " ", sent).strip()
                    if not clean or clean in docs[norm_title]["seen_sentences"]:
                        continue
                    docs[norm_title]["sentences"].append(clean)
                    docs[norm_title]["seen_sentences"].add(clean)

        self._index = []
        self._full = {}
        self._title_to_id = {}
        for idx, payload in enumerate(docs.values(), start=1):
            text = " ".join(payload["sentences"])
            doc_id = f"doc_{idx:05d}"
            entry = {
                "doc_id": doc_id,
                "title": payload["title"],
                "text": text,
            }
            self._index.append(entry)
            self._full[doc_id] = text
            self._full[payload["title"]] = text
            self._title_to_id[_normalize(payload["title"])] = doc_id

        self._rebuild_inverted()

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                for entry in self._index:
                    f.write(json.dumps(entry) + "\n")
            print(f"Tool index built: {len(self._index)} documents → {index_path}")

    def build_from_traces(self, jsonl_path: str, index_path: Optional[str] = None) -> None:
        """
        Build a lightweight index from the existing SFT trace JSONL file by
        extracting the document snippets embedded in read steps. Useful when
        the full HotpotQA dataset is not available.
        """
        seen: set[str] = set()
        self._index = []
        self._full = {}
        self._title_to_id = {}
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                for step in rec.get("trace", []):
                    if step.get("action") == "read":
                        doc = step.get("document", "")
                        key = doc[:60]
                        if key not in seen:
                            seen.add(key)
                            doc_id = f"doc_{len(self._index) + 1:05d}"
                            entry = {"doc_id": doc_id, "title": key, "text": doc}
                            self._index.append(entry)
                            self._full[doc_id] = doc
                            self._full[key] = doc
                            self._title_to_id[_normalize(key)] = doc_id

        self._rebuild_inverted()

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                for entry in self._index:
                    f.write(json.dumps(entry) + "\n")
        print(f"Tool index built from traces: {len(self._index)} documents")

    def _add_to_inverted(self, entry: dict, idx: int) -> None:
        tokens = set(_tokenize(entry["title"] + " " + entry["text"]))
        for tok in tokens:
            self._inv.setdefault(tok, []).append(idx)

    def _rebuild_inverted(self) -> None:
        self._inv = {}
        self._term_freqs = []
        self._title_terms = []
        self._doc_freq = {}
        self._doc_len = []

        for idx, entry in enumerate(self._index):
            self._add_to_inverted(entry, idx)
            doc_tokens = _tokenize(entry["title"] + " " + entry["text"])
            self._term_freqs.append(dict(Counter(doc_tokens)))
            self._title_terms.append(set(_tokenize(entry["title"])))
            self._doc_len.append(len(doc_tokens))
            for tok in set(doc_tokens):
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1

        self._avg_doc_len = (
            sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        )

    def _bm25_score(self, query: str, idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """BM25-style score with light title boosting for entity-heavy queries."""
        if not self._index:
            return 0.0

        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        tf = self._term_freqs[idx]
        title_terms = self._title_terms[idx]
        dl = max(self._doc_len[idx], 1)
        avgdl = max(self._avg_doc_len, 1.0)
        n_docs = max(len(self._index), 1)
        score = 0.0

        for tok, qtf in Counter(query_tokens).items():
            term_tf = tf.get(tok, 0)
            if term_tf == 0:
                continue

            df = self._doc_freq.get(tok, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = term_tf + k1 * (1 - b + b * (dl / avgdl))
            score += qtf * idf * (term_tf * (k1 + 1)) / max(denom, 1e-8)

            if tok in title_terms:
                score += 1.25 * idf

        norm_query = _normalize(query)
        norm_title = _normalize(self._index[idx]["title"])
        if norm_query == norm_title:
            score += 5.0
        elif norm_query in norm_title:
            score += 2.5

        return score

    def _load_index(self, path: str) -> None:
        self._index = []
        self._full = {}
        self._title_to_id = {}
        with open(path) as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line.strip())
                if "doc_id" not in entry:
                    entry["doc_id"] = f"doc_{i:05d}"
                self._index.append(entry)
                self._full[entry["doc_id"]] = entry["text"]
                self._full[entry["title"]] = entry["text"]
                self._title_to_id[_normalize(entry["title"])] = entry["doc_id"]
        self._rebuild_inverted()
        print(f"Tool index loaded: {len(self._index)} documents from {path}")

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """
        Return the top-k most relevant documents for the query.

        Output format is intentionally compact and tool-friendly:
            [doc_00042] Albert Einstein :: Albert Einstein was born in Ulm...

        This makes it easy for the model to pass either the `doc_id` or title
        into `read()` on the next step.

        Uses inverted index to avoid O(N) linear scan.
        """
        if not self._index:
            return "[No index loaded. Call build_from_hotpotqa() first.]"

        query_tokens = set(_normalize(query).split())
        if not query_tokens:
            return f"[Empty query]"

        # Collect candidate documents that share at least one token.
        candidate_ids: dict[int, int] = {}  # idx -> shared token count
        for tok in query_tokens:
            for idx in self._inv.get(tok, []):
                candidate_ids[idx] = candidate_ids.get(idx, 0) + 1

        if not candidate_ids:
            return f"[No results found for: {query}]"

        # Score candidates with BM25-style retrieval plus title boost.
        scored = []
        for idx, _hits in candidate_ids.items():
            entry = self._index[idx]
            score = self._bm25_score(query, idx)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]

        if not top or top[0][0] == 0.0:
            return f"[No results found for: {query}]"

        parts = []
        for score, entry in top:
            snippet = entry["text"][:220].rstrip()
            if len(entry["text"]) > 220:
                snippet += "..."
            parts.append(f"[{entry['doc_id']}] {entry['title']} :: {snippet}")
        return "\n\n".join(parts)

    def read(self, doc_ref: str) -> str:
        """
        Return the full document for a reference.

        Accepted references:
        - `doc_id` from `search()` output, e.g. `doc_00042`
        - exact title
        - search result line containing `[doc_id] Title :: snippet`
        - approximate snippet text from older SFT traces
        """
        if not self._index:
            return "[No index loaded.]"

        doc_ref = doc_ref.strip()
        if not doc_ref:
            return "[Empty document reference.]"

        # Exact match by doc_id or title.
        if doc_ref in self._full:
            return self._full[doc_ref]

        # Extract doc_id from formatted search result line.
        m = re.search(r"\[(doc_\d+)\]", doc_ref)
        if m and m.group(1) in self._full:
            return self._full[m.group(1)]

        # Try exact/normalized title lookup from a full search result line.
        title_part = doc_ref
        if "] " in doc_ref:
            title_part = doc_ref.split("] ", 1)[1]
        if " :: " in title_part:
            title_part = title_part.split(" :: ", 1)[0]
        norm_title = _normalize(title_part)
        doc_id = self._title_to_id.get(norm_title)
        if doc_id and doc_id in self._full:
            return self._full[doc_id]

        # Otherwise find the document with highest overlap to the reference text.
        scored = [
            (_token_overlap(doc_ref, e["title"] + " " + e["text"]), e)
            for e in self._index
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0.1:
            return scored[0][1]["text"]

        return f"[Could not find document for reference: {doc_ref[:80]}...]"
