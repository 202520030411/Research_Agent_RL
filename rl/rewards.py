"""
Reward computation for the RL stopping policy.

Reward formula (from the proposal):
    R = +1  [correct answer]
      - α·t  [step penalty]
      - β·Σ_t 1[JSD(P_ans^t ∥ P_ans^{t-1}) < ε]  [low-info-gain penalty]

The JSD signal uses the model's confidence output as a proxy for the
answer distribution P_ans^t. Each confidence value is treated as a
Bernoulli(p) distribution where p = P(answer is correct at step t).

JSD between Bernoulli(p) and Bernoulli(q):
    m = (p + q) / 2
    JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)

If JSD < ε, the step added negligible information → penalise.

Hyperparameters are read from config.yaml but can be overridden at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# JSD utilities
# ---------------------------------------------------------------------------

def _kl_bernoulli(p: float, q: float, eps: float = 1e-8) -> float:
    """KL(Bernoulli(p) || Bernoulli(q))."""
    p = max(min(p, 1 - eps), eps)
    q = max(min(q, 1 - eps), eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def jsd_bernoulli(p: float, q: float) -> float:
    """
    Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).
    Returns a value in [0, log(2)] ≈ [0, 0.693].
    """
    m = (p + q) / 2.0
    return 0.5 * _kl_bernoulli(p, m) + 0.5 * _kl_bernoulli(q, m)


def low_info_gain_steps(confidences: list[float], epsilon: float) -> int:
    """
    Count the number of consecutive-step pairs where JSD < epsilon.
    Pairs: (conf[0], conf[1]), (conf[1], conf[2]), ...
    """
    count = 0
    for i in range(1, len(confidences)):
        if jsd_bernoulli(confidences[i], confidences[i - 1]) < epsilon:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Reward dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Full breakdown of the reward for one episode."""
    correctness:   float   # +1 or 0
    step_penalty:  float   # -α * t
    jsd_penalty:   float   # -β * count_low_info_steps
    total:         float   # sum of above

    def __repr__(self) -> str:
        return (
            f"R={self.total:+.3f}  "
            f"(corr={self.correctness:+.1f}, "
            f"step={self.step_penalty:+.3f}, "
            f"jsd={self.jsd_penalty:+.3f})"
        )


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    correct: bool,
    n_steps: int,
    confidences: list[float],
    alpha: float = 0.1,
    beta:  float = 0.05,
    epsilon: float = 0.05,
) -> RewardBreakdown:
    """
    Compute the full reward for one agent episode.

    Args:
        correct     : whether the agent's final answer was correct
        n_steps     : total number of LLM calls in the episode
        confidences : list of confidence values at each step
        alpha       : weight for the step-count penalty
        beta        : weight for the low-info-gain penalty
        epsilon     : JSD threshold below which a step is penalised

    Returns:
        RewardBreakdown with per-component and total reward
    """
    r_correctness  = 1.0 if correct else 0.0
    r_step_penalty = -alpha * n_steps
    n_low_info     = low_info_gain_steps(confidences, epsilon)
    r_jsd_penalty  = -beta * n_low_info

    total = r_correctness + r_step_penalty + r_jsd_penalty

    return RewardBreakdown(
        correctness=r_correctness,
        step_penalty=r_step_penalty,
        jsd_penalty=r_jsd_penalty,
        total=total,
    )


def compute_reward_from_result(result, alpha: float, beta: float, epsilon: float) -> RewardBreakdown:
    """
    Convenience wrapper: compute reward directly from an AgentResult object.
    """
    confidences = [rec.confidence for rec in result.steps]
    return compute_reward(
        correct=result.correct,
        n_steps=result.n_steps,
        confidences=confidences,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
    )
