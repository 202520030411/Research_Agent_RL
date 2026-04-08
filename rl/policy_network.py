"""
RL Policy Network: lightweight MLP for the stopping decision.

Input features (4-dimensional):
    [step_count_norm, last_confidence, avg_confidence, tool_call_count_norm]

Output:
    Scalar in (0, 1) — probability of stopping at this step.
    The policy samples a Bernoulli(p_stop) action at each step.

The network is deliberately tiny (two hidden layers, 64 units each) so it
can be trained on a single GPU with REINFORCE in under an hour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


FEATURE_DIM  = 4   # [step_norm, last_conf, avg_conf, tool_norm]
MAX_STEPS    = 8   # used for normalisation
MAX_TOOLS    = 8


def make_features(
    step_count: int,
    last_confidence: float,
    avg_confidence: float,
    tool_call_count: int,
) -> torch.Tensor:
    """
    Build a (1, 4) feature tensor from scalar agent-state values.
    All features are normalised to roughly [0, 1].
    """
    return torch.tensor([[
        step_count       / MAX_STEPS,
        last_confidence,              # already in [0, 1]
        avg_confidence,               # already in [0, 1]
        tool_call_count  / MAX_TOOLS,
    ]], dtype=torch.float32)


def features_from_history(history: list) -> torch.Tensor:
    """
    Convenience wrapper: build features directly from a list of StepRecord.
    """
    if not history:
        return make_features(0, 0.0, 0.0, 0)

    step_count  = len(history)
    tool_count  = sum(1 for r in history if r.action in ("search", "read"))
    last_conf   = history[-1].confidence
    avg_conf    = sum(r.confidence for r in history) / step_count

    return make_features(step_count, last_conf, avg_conf, tool_count)


class PolicyNetwork(nn.Module):
    """
    Two-layer MLP that maps agent state features to P(stop).

    Architecture:
        Linear(4 → 64) → LayerNorm → ReLU
        Linear(64 → 64) → LayerNorm → ReLU
        Linear(64 → 1)  → Sigmoid

    LayerNorm instead of BatchNorm because batch size is often 1 at
    inference time.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4) → (B, 1) probability of stopping."""
        return self.net(x)

    def stop_prob(self, history: list) -> float:
        """Return scalar P(stop) for a given history of StepRecords."""
        feats = features_from_history(history).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.forward(feats).item()

    def sample_action(self, history: list) -> tuple[int, torch.Tensor]:
        """
        Sample a binary action: 1 = stop, 0 = continue.

        Returns:
            action  : int (0 or 1)
            log_prob: scalar tensor (for REINFORCE gradient)
        """
        feats    = features_from_history(history).to(next(self.parameters()).device)
        p_stop   = self.forward(feats).squeeze()          # scalar
        dist     = torch.distributions.Bernoulli(p_stop)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, hidden: int = 64) -> "PolicyNetwork":
        net = cls(hidden=hidden)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return net
