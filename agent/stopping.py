"""
Stopping policies for the research agent.

Each policy implements a single method:
    should_stop(history: list[StepRecord]) -> bool

Three policies are provided:
  1. NeverStop           — always continue (oracle upper bound on steps)
  2. FixedStepPolicy     — stop after exactly N tool-call steps
  3. ConfidencePolicy    — stop when confidence >= threshold
  4. RLPolicy            — placeholder; wraps the Week-3 MLP (not trained yet)

The agent calls should_stop() BEFORE each LLM call. If True, the agent
requests one final answer step from the model and terminates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import StepRecord


class StoppingPolicy(ABC):
    """Base class for all stopping policies."""

    @abstractmethod
    def should_stop(self, history: list["StepRecord"]) -> bool:
        """Return True if the agent should stop before the next LLM call."""

    def reset(self) -> None:
        """Called at the start of each new question (for stateful policies)."""


# ---------------------------------------------------------------------------
# Baseline 0 — never stop (let max_steps handle termination)
# ---------------------------------------------------------------------------

class NeverStop(StoppingPolicy):
    """Never triggers early stopping. The agent runs until max_steps or answer."""

    def should_stop(self, history: list["StepRecord"]) -> bool:
        return False

    def __repr__(self) -> str:
        return "NeverStop()"


# ---------------------------------------------------------------------------
# Baseline 1 — fixed number of steps
# ---------------------------------------------------------------------------

class FixedStepPolicy(StoppingPolicy):
    """
    Stop after exactly `max_steps` tool-call steps have been completed.

    A 'tool-call step' is any step with action in {"search", "read"}.
    The answer step itself does not count.

    Args:
        max_steps : number of tool calls before stopping (default 3)
    """

    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps

    def should_stop(self, history: list["StepRecord"]) -> bool:
        tool_calls = sum(
            1 for rec in history if rec.action in ("search", "read")
        )
        return tool_calls >= self.max_steps

    def __repr__(self) -> str:
        return f"FixedStepPolicy(max_steps={self.max_steps})"


# ---------------------------------------------------------------------------
# Baseline 2 — confidence threshold
# ---------------------------------------------------------------------------

class ConfidencePolicy(StoppingPolicy):
    """
    Stop when the model's most recent confidence score exceeds `threshold`.

    Args:
        threshold   : confidence value in [0, 1] at which to stop (default 0.8)
        min_steps   : minimum tool calls before the policy can trigger (default 1)
                      Prevents stopping before any information is gathered.
    """

    def __init__(self, threshold: float = 0.8, min_steps: int = 1):
        self.threshold = threshold
        self.min_steps = min_steps

    def should_stop(self, history: list["StepRecord"]) -> bool:
        if not history:
            return False

        tool_calls = sum(
            1 for rec in history if rec.action in ("search", "read")
        )
        if tool_calls < self.min_steps:
            return False

        latest_confidence = history[-1].confidence
        return latest_confidence >= self.threshold

    def __repr__(self) -> str:
        return f"ConfidencePolicy(threshold={self.threshold}, min_steps={self.min_steps})"


# ---------------------------------------------------------------------------
# Week-3 placeholder — RL policy
# ---------------------------------------------------------------------------

class RLPolicy(StoppingPolicy):
    """
    Stopping policy backed by the Week-3 MLP trained with PPO/REINFORCE.

    The MLP takes a feature vector:
        [step_count, last_confidence, avg_confidence, tool_call_count]
    and outputs a binary {continue=0, stop=1} decision.

    Args:
        model_path : path to saved MLP weights (set in Week 3)
        threshold  : probability threshold above which to stop (default 0.5)
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        policy_network=None,
    ):
        self.threshold = threshold
        self.mlp       = policy_network  # accept a pre-loaded PolicyNetwork directly

        if model_path and self.mlp is None:
            self._load(model_path)

    def _load(self, path: str) -> None:
        import torch
        from rl.policy_network import PolicyNetwork  # imported only when available
        self.mlp = PolicyNetwork.load(path)
        self.mlp.eval()

    def _features(self, history: list["StepRecord"]) -> list[float]:
        tool_steps = [r for r in history if r.action in ("search", "read")]
        step_count  = len(history)
        tool_count  = len(tool_steps)
        last_conf   = history[-1].confidence if history else 0.0
        avg_conf    = (sum(r.confidence for r in history) / len(history)
                       if history else 0.0)
        return [float(step_count), last_conf, avg_conf, float(tool_count)]

    def should_stop(self, history: list["StepRecord"]) -> bool:
        if self.mlp is None or not history:
            return False

        import torch
        feats  = torch.tensor([self._features(history)], dtype=torch.float32)
        with torch.no_grad():
            prob_stop = self.mlp(feats).item()
        return prob_stop >= self.threshold

    def __repr__(self) -> str:
        loaded = self.mlp is not None
        return f"RLPolicy(loaded={loaded}, threshold={self.threshold})"
