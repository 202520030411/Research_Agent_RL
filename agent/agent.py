"""
Core research agent interaction loop.

The agent runs a question through the SFT LLM, parses the structured JSON
output, executes the requested tool, feeds the result back as context, and
repeats until a stopping policy decides to halt or the model outputs an answer.

Usage:
    agent = ResearchAgent(model, tokenizer, tool_executor, stopping_policy)
    result = agent.run(question)
"""

import json
import re
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from agent.tools import ToolExecutor
from agent.stopping import StoppingPolicy
from data.sft_dataset import SYSTEM_PROMPT

MAX_NEW_TOKENS = 400
TEMPERATURE    = 0.1


@dataclass
class StepRecord:
    """One step in the agent's reasoning trace."""
    step_idx:    int
    action:      str          # "search" | "read" | "answer"
    thought:     str
    confidence:  float
    tool_input:  str = ""    # query for search, document snippet for read
    tool_output: str = ""    # what the tool returned
    answer:      str = ""    # populated on action="answer"
    raw_output:  str = ""    # full raw model output for debugging


@dataclass
class AgentResult:
    question:     str
    gold_answer:  str
    pred_answer:  str
    steps:        list[StepRecord] = field(default_factory=list)
    n_steps:      int = 0
    n_tool_calls: int = 0
    correct:      bool = False
    stopped_by:   str = ""   # "policy" | "answer" | "max_steps" | "parse_error"


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_first_json(text: str) -> dict | None:
    """
    Scan character by character to find the first balanced {...} block
    and parse it as JSON. More reliable than a greedy regex.
    """
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1  # reset and keep looking
    return None


def _parse_step(text: str) -> dict | None:
    """
    Robustly extract the first valid JSON object from the model output.

    Strategies (tried in order):
      1. Strip 'Step N:' prefix, scan for balanced braces.
      2. Try each line individually.
      3. Try json.loads on the whole stripped text.
    """
    if not text or not text.strip():
        return None

    # Strip 'Step N:' prefix if present (only from the very start)
    text = re.sub(r"^Step\s*\d+\s*:\s*", "", text.strip())

    parsed = _extract_first_json(text)
    if parsed is not None:
        return parsed

    for line in text.splitlines():
        line = re.sub(r"^Step\s*\d+\s*:\s*", "", line.strip())
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_answer_from_raw(text: str) -> str:
    """
    Last-resort: try to find an answer string in raw unparsed model output.
    Looks for `"answer": "..."` key-value pair directly.
    """
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return ""


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------

def _normalize_answer(ans: str) -> str:
    return ans.strip().lower().rstrip(".")


def _is_correct(pred: str, gold: str) -> bool:
    if not pred or not pred.strip():
        return False
    p = _normalize_answer(pred)
    g = _normalize_answer(gold)
    if not p or not g:
        return False
    return p == g or g in p or p in g


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Runs the search-read-answer loop for a single question.

    Prompt design matches the SFT training format exactly:
      The model always sees [system, user(question), assistant(partial trace)]
      and generates the NEXT step to append to the partial trace.
      Tool observations are injected as "Observation: ..." lines inside the
      assistant turn — never as separate user messages.

    Args:
        model           : loaded PeftModel (SFT fine-tuned)
        tokenizer       : matching tokenizer
        tool_executor   : ToolExecutor instance
        stopping_policy : StoppingPolicy instance
        max_steps       : hard cap on number of LLM calls (excluding the bonus
                          answer call made after a policy stop)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tool_executor: ToolExecutor,
        stopping_policy: StoppingPolicy,
        max_steps: int = 8,
    ):
        self.model           = model
        self.tokenizer       = tokenizer
        self.tool_executor   = tool_executor
        self.stopping_policy = stopping_policy
        self.max_steps       = max_steps
        self.device          = next(model.parameters()).device

    def _build_prompt(self, question: str, history: list[StepRecord]) -> str:
        """
        Build the generation prompt.

        For step 1 (empty history):
            <system> ... <user> Question: ... <|im_start|>assistant\n

        For step N (history non-empty):
            <system> ... <user> Question: ... <|im_start|>assistant\n
            Step 1: {...}\nObservation: ...\nStep 2: {...}\n...
            Step N: [model generates here]

        Prior steps + observations are appended as partial assistant content
        so the model continues its own turn — matching the SFT training format.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}"},
        ]
        # Base prompt ends with <|im_start|>assistant\n
        base = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if not history:
            return base

        # Append prior steps and observations as partial assistant content
        lines = []
        for rec in history:
            step_json: dict = {
                "thought":    rec.thought,
                "action":     rec.action,
                "confidence": rec.confidence,
            }
            if rec.action == "search":
                step_json["query"] = rec.tool_input
            elif rec.action == "read":
                step_json["document"] = rec.tool_input
            elif rec.action == "answer":
                step_json["answer"] = rec.answer

            lines.append(f"Step {rec.step_idx + 1}: {json.dumps(step_json)}")
            if rec.tool_output:
                lines.append(f"Observation: {rec.tool_output[:300]}")

        # Prompt the model to generate the next step
        next_step_idx = len(history) + 1
        lines.append(f"Step {next_step_idx}: ")

        return base + "\n".join(lines)

    @torch.no_grad()
    def _call_llm(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def run(self, question: str, gold_answer: str = "") -> AgentResult:
        result = AgentResult(
            question=question, gold_answer=gold_answer, pred_answer=""
        )
        history: list[StepRecord] = []

        for step_idx in range(self.max_steps):
            # Check stopping policy before calling LLM
            if self.stopping_policy.should_stop(history):
                result.stopped_by = "policy"
                # One final LLM call to get the answer given accumulated context
                raw = self._call_llm(self._build_prompt(question, history))
                parsed = _parse_step(raw)
                if parsed and parsed.get("action") == "answer":
                    result.pred_answer = parsed.get("answer", "")
                elif not result.pred_answer:
                    result.pred_answer = _extract_answer_from_raw(raw)
                result.n_steps += 1  # count the bonus answer call
                break

            prompt     = self._build_prompt(question, history)
            raw_output = self._call_llm(prompt)
            parsed     = _parse_step(raw_output)

            if parsed is None:
                # Try to salvage an answer from the raw text before giving up
                salvaged = _extract_answer_from_raw(raw_output)
                if salvaged:
                    result.pred_answer = salvaged
                result.stopped_by = "parse_error"
                result.n_steps    = len(history)
                break

            action     = parsed.get("action", "answer")
            thought    = parsed.get("thought", "")
            confidence = float(parsed.get("confidence", 0.5))

            rec = StepRecord(
                step_idx=step_idx,
                action=action,
                thought=thought,
                confidence=confidence,
                raw_output=raw_output,
            )

            if action == "answer":
                rec.answer         = parsed.get("answer", "")
                result.pred_answer = rec.answer
                result.stopped_by  = "answer"
                history.append(rec)
                break

            elif action == "search":
                query           = parsed.get("query", question)
                rec.tool_input  = query
                rec.tool_output = self.tool_executor.search(query)
                result.n_tool_calls += 1

            elif action == "read":
                doc             = parsed.get("document", "")
                rec.tool_input  = doc
                rec.tool_output = self.tool_executor.read(doc)
                result.n_tool_calls += 1

            history.append(rec)

        else:
            result.stopped_by = "max_steps"
            if not result.pred_answer and history:
                result.pred_answer = history[-1].answer or ""

        result.steps   = history
        result.n_steps = len(history)
        result.correct = _is_correct(result.pred_answer, gold_answer)
        return result
