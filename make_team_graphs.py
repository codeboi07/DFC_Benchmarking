"""Publication-ready baseline-vs-DFC graphs (AgentLeak style) for GPT-OSS 120B.

Numbers (from the verified full 560/560 run):
  baseline utility 41.8%, ASR 24.3%  ->  resistance 75.7%
  DFC      utility 56.2%, ASR  3.9%  ->  resistance 96.1%
  classification: Type1 caught 118, Type2 false-pos 69, Type3 missed 22, clean 355
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results/dfc_comparison")
OUT.mkdir(parents=True, exist_ok=True)

GRAY = "#9AA5B1"   # baseline
BLUE = "#1F6FEB"   # DFC
TITLE_FS, LABEL_FS, TICK_FS, VAL_FS, LEG_FS = 15, 13, 12, 12, 12


def clean(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(False)


# ───────────────────────── Graph 1: utility & security (resistance) ─────────────────────────
baseline = {"Utility": 41.8, "Security\n(resistance)": 75.7}
dfc = {"Utility": 56.2, "Security\n(resistance)": 96.1}
labels = list(baseline.keys())
x = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
b1 = ax.bar(x - w / 2, list(baseline.values()), w, label="Baseline", color=GRAY, edgecolor="white", linewidth=1)
b2 = ax.bar(x + w / 2, list(dfc.values()), w, label="DFC", color=BLUE, edgecolor="white", linewidth=1)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=VAL_FS, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=LABEL_FS)
ax.set_ylabel("Rate (%)  —  higher is better", fontsize=LABEL_FS)
ax.set_title("GPT-OSS 120B: Baseline vs DFC\n(workspace, important_instructions)", fontsize=TITLE_FS, fontweight="bold")
ax.legend(fontsize=LEG_FS, frameon=False, loc="upper left")
clean(ax)
plt.tight_layout()
p1 = OUT / "gptoss_team_utility_security.png"
fig.savefig(p1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p1}")

# ───────────────────────── Graph 2: outcome classification ─────────────────────────
cats = ["Type 1\ncaught", "Type 2\nfalse positive", "Type 3\nmissed", "Clean\n(no attack)"]
vals = [118, 69, 22, 355]
colors = ["#2E7D32", "#F2A900", "#C62828", "#9AA5B1"]  # good / bad / bad / neutral

fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="white")
bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=1, width=0.62)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(v),
            ha="center", va="bottom", fontsize=VAL_FS, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.12)
ax.set_ylabel("Number of (user task × injection) pairs", fontsize=LABEL_FS)
ax.set_title("DFC Outcome Classification — GPT-OSS 120B\n(workspace, important_instructions, n=560)",
             fontsize=TITLE_FS, fontweight="bold")
clean(ax)
plt.tight_layout()
p2 = OUT / "gptoss_team_classification.png"
fig.savefig(p2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p2}")
