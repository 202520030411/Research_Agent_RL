"""
Render the two report figures from the experimental data.

Figure 1: bar chart of exact-match across the 7 evaluation configurations
          (5 Week 2 baselines + SFT-only-at-max_steps=5 + GRPO).
Figure 2: GRPO validation accuracy across 200 training steps, with the
          approximate standard-error band for the 20-question samples,
          and the 100-question final eval as a dashed reference line.

Outputs: report/figures/fig1_main.{png,pdf}, report/figures/fig2_val.{png,pdf}
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Uniform paper style (no emojis, serif font to match LaTeX article class).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Figure 1: main result bar chart
# ---------------------------------------------------------------------------

# Bars in the order they appear in Table 1.
bars = [
    ("FixedStep(N=2)",        6.0,  None,  "baseline"),
    ("FixedStep(N=3)",        16.0, None,  "baseline"),
    (r"Confidence($\tau=0.75$)",  20.0, None, "baseline"),
    (r"Confidence($\tau=0.85$)",  26.0, None, "baseline"),
    (r"NeverStop(max=6)",     21.0, None,  "baseline"),
    ("SFT-only\n(max_steps=5)", 43.0, (1.0, 1.0), "sft"),  # midpoint 43, +/-1 for 42-44 range
    ("Multi-turn GRPO\n(ours)", 44.0, None, "grpo"),
]

COLORS = {
    "baseline": "#bdbdbd",  # light grey
    "sft":      "#6baed6",  # muted blue
    "grpo":     "#08519c",  # strong blue
}

fig, ax = plt.subplots(figsize=(7.0, 3.8))
x = np.arange(len(bars))
heights = [b[1] for b in bars]
colors  = [COLORS[b[3]] for b in bars]

# Custom error bars for SFT-only row only (shows two-run spread).
yerr = np.zeros((2, len(bars)))
for i, b in enumerate(bars):
    if b[2] is not None:
        yerr[0, i], yerr[1, i] = b[2]

rects = ax.bar(
    x, heights,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
    yerr=yerr if yerr.any() else None,
    error_kw=dict(ecolor="black", capsize=4, elinewidth=1.0),
)

# Value labels above each bar.
for i, (rect, b) in enumerate(zip(rects, bars)):
    if b[2] is not None:
        label = f"{b[1]:.1f}\n(42.0 – 44.0)"
    else:
        label = f"{b[1]:.1f}"
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + (1.5 if b[2] is None else 2.5),
        label,
        ha="center", va="bottom", fontsize=8,
    )

# Visual break between "zero-shot baselines" and "5-step" configurations.
ax.axvline(4.5, linestyle=":", color="grey", alpha=0.6, linewidth=0.8)
ax.text(2.25, 55, "zero-shot stopping heuristics",
        ha="center", fontsize=8, style="italic", color="dimgrey")
ax.text(5.5, 55, "full 5-step trajectory",
        ha="center", fontsize=8, style="italic", color="dimgrey")

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], rotation=25, ha="right")
ax.set_ylabel("Exact-match accuracy (%)")
ax.set_title("HotpotQA validation (n=100): decomposition of the 26%→44% gap",
             pad=12)
ax.set_ylim(0, 60)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(FIG_DIR / "fig1_main.png")
fig.savefig(FIG_DIR / "fig1_main.pdf")
print(f"  wrote {FIG_DIR / 'fig1_main.png'}")
print(f"  wrote {FIG_DIR / 'fig1_main.pdf'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: GRPO validation curve
# ---------------------------------------------------------------------------

epochs = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
val    = np.array([0.50, 0.45, 0.35, 0.35, 0.40, 0.45, 0.40, 0.50, 0.40, 0.50])

# Wald SE at each point: sqrt(p*(1-p)/n) with n=20.
se = np.sqrt(val * (1 - val) / 20.0)

final_100 = 0.44

fig, ax = plt.subplots(figsize=(6.5, 3.6))

ax.fill_between(
    epochs, val - se, val + se,
    color="#9ecae1", alpha=0.35, label=r"$\pm 1$ SE (n=20 per checkpoint)",
)
ax.plot(
    epochs, val,
    color="#08519c", marker="o", markersize=5, linewidth=1.5,
    label="[VAL] exact-match (20-q subsample)",
)
ax.axhline(
    final_100, linestyle="--", color="crimson", linewidth=1.2,
    label=f"100-question final eval ({final_100:.2f})",
)

ax.set_xlabel("GRPO update step")
ax.set_ylabel("Exact-match accuracy")
ax.set_title("GRPO validation accuracy across 200 update steps",
             pad=10)
ax.set_xticks(epochs)
ax.set_ylim(0.20, 0.70)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", framealpha=0.9)

fig.savefig(FIG_DIR / "fig2_val.png")
fig.savefig(FIG_DIR / "fig2_val.pdf")
print(f"  wrote {FIG_DIR / 'fig2_val.png'}")
print(f"  wrote {FIG_DIR / 'fig2_val.pdf'}")
plt.close(fig)

print("done.")
