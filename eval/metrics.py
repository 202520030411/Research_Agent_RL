"""
Evaluation metrics for the research agent.

compute_metrics(results) -> dict  given a list of AgentResult objects.

Metrics:
  accuracy          : fraction of questions answered correctly
  avg_steps         : mean number of LLM calls per question
  avg_tool_calls    : mean number of tool calls (search + read) per question
  efficiency_score  : accuracy / avg_tool_calls  (higher = better)
  stopped_by_*      : fraction of episodes stopped by each mechanism
"""

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentResult


def compute_metrics(results: list["AgentResult"]) -> dict:
    """
    Args:
        results : list of AgentResult objects from ResearchAgent.run()

    Returns:
        dict with scalar metric values
    """
    n = len(results)
    if n == 0:
        return {}

    n_correct    = sum(r.correct for r in results)
    total_steps  = sum(r.n_steps for r in results)
    total_tools  = sum(r.n_tool_calls for r in results)
    stop_counts  = Counter(r.stopped_by for r in results)

    accuracy        = n_correct / n
    avg_steps       = total_steps / n
    avg_tool_calls  = total_tools / n
    efficiency      = accuracy / avg_tool_calls if avg_tool_calls > 0 else 0.0

    metrics = {
        "n_questions":      n,
        "accuracy":         round(accuracy, 4),
        "avg_steps":        round(avg_steps, 2),
        "avg_tool_calls":   round(avg_tool_calls, 2),
        "efficiency_score": round(efficiency, 4),
    }

    # Stopped-by breakdown
    for reason, count in stop_counts.items():
        metrics[f"stopped_by_{reason}"] = round(count / n, 3)

    return metrics


def format_metrics(metrics: dict, policy_name: str = "") -> str:
    """Pretty-print metrics as a table row."""
    label = f"{policy_name:30s}" if policy_name else ""
    return (
        f"{label}"
        f"acc={metrics.get('accuracy', 0):.3f}  "
        f"steps={metrics.get('avg_steps', 0):.1f}  "
        f"tools={metrics.get('avg_tool_calls', 0):.1f}  "
        f"eff={metrics.get('efficiency_score', 0):.3f}"
    )


def compare_policies(results_by_policy: dict[str, list["AgentResult"]]) -> str:
    """
    Given {policy_name: [AgentResult, ...]}, return a formatted comparison table.
    """
    lines = [
        f"{'Policy':<30} {'Accuracy':>8} {'Avg Steps':>10} {'Avg Tools':>10} {'Efficiency':>11}",
        "-" * 65,
    ]
    for name, results in results_by_policy.items():
        m = compute_metrics(results)
        lines.append(
            f"{name:<30} {m['accuracy']:>8.3f} {m['avg_steps']:>10.1f} "
            f"{m['avg_tool_calls']:>10.1f} {m['efficiency_score']:>11.4f}"
        )
    return "\n".join(lines)
