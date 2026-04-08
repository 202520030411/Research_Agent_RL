"""
Stopping policies for the research agent.

Each policy implements a single method:
    should_stop(history: list[StepRecord]) -> bool

Three policies are provided:
  1. NeverStop           — always continue (oracle upper bound on steps)
  2. FixedStepPolicy     — stop after exactly N tool-call steps
  3. ConfidencePolicy    — stop when confidence >= threshold

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
