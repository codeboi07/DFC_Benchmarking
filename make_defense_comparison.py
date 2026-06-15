"""GPT-OSS 120B defense comparison (workspace, important_instructions).

Computes utility/ASR/resistance for each setting directly from runs/ (no hardcoded numbers),
prints a table, and produces a grouped-bar chart + a utility-vs-resistance scatter.
security==True => attack SUCCEEDED. ASR = mean(security==True). resistance = 100-ASR.
tool_filter is omitted: it is OpenAI-API-only and cannot run on an open-weight Bedrock agent.
"""

import json
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/dfc_comparison")
OUT.mkdir(parents=True, exist_ok=True)

# label -> run dir
SETTINGS = [
    ("Baseline", "runs/gpt-oss-120b"),
    ("spotlighting", "runs/gpt-oss-120b-spotlighting_with_delimiting"),
    ("repeat_user_prompt", "runs/gpt-oss-120b-repeat_user_prompt"),
    ("pi_detector", "runs/gpt-oss-120b-transformers_pi_detector"),
    ("DFC", "runs/gpt-oss-120b-dfc_agent_framework_integration"),
]


def nums(path):
    files = [f for f in glob.glob(path + "/workspace/user_task_*/important_instructions/injection_task_*.json")
             if "_dfc" not in f and "metadata" not in f]
    u = s = n = 0
    for f in files:
        d = json.load(open(f))
        if d.get("attack_type") != "important_instructions" or d.get("injection_task_id") is None:
            continue
        if d.get("security") is None:
            continue
        n += 1
        u += d["utility"] is True
        s += d["security"] is True
    return n, 100 * u / n, 100 * s / n


rows = []
for label, path in SETTINGS:
    n, util, asr = nums(path)
    rows.append((label, n, util, asr, 100 - asr))

print("=" * 78)
print("GPT-OSS 120B — Defense Comparison (workspace, important_instructions)".center(78))
print("CONVENTION: ASR = mean(security==True), lower=better. resistance=100-ASR.".center(78))
print("=" * 78)
print(f"{'setting':<22}{'n':>6}{'utility%':>11}{'ASR%':>9}{'resistance%':>14}")
for label, n, util, asr, res in rows:
    print(f"{label:<22}{n:>6}{util:>11.1f}{asr:>9.1f}{res:>14.1f}")
print(f"{'tool_filter':<22}{'—':>6}{'N/A — OpenAI-API-only (incompatible with open-weight Bedrock)':>1}")
print("=" * 78)

labels = [r[0] for r in rows]
util = [r[2] for r in rows]
asr = [r[3] for r in rows]
res = [r[4] for r in rows]

# ---------- Grouped bar: utility + resistance (both higher=better) ----------
x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")
b1 = ax.bar(x - w / 2, util, w, label="Utility", color="#1F6FEB", edgecolor="white")
b2 = ax.bar(x + w / 2, res, w, label="Resistance (100−ASR)", color="#2E7D32", edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 108)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Rate (%)  —  higher is better", fontsize=13)
ax.set_title("GPT-OSS 120B: Defense Comparison\n(workspace, important_instructions, n=560)",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=12, frameon=False, loc="upper left")
for spn in ("top", "right"):
    ax.spines[spn].set_visible(False)
ax.grid(False)
plt.tight_layout()
p1 = OUT / "gptoss_defense_comparison.png"
fig.savefig(p1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p1}")

# ---------- Scatter: utility vs resistance (DFC should be top-right) ----------
colors = ["#9AA5B1", "#9AA5B1", "#9AA5B1", "#F2A900", "#1F6FEB"]
fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="white")
for (label, _n, u, a, r), c in zip(rows, colors):
    ax.scatter(u, r, s=240, color=c, edgecolor="black", linewidth=1.2, zorder=3)
    ax.annotate(label, (u, r), textcoords="offset points", xytext=(9, 5), fontsize=12, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(50, 102)
ax.set_xlabel("Utility (%)  —  higher better", fontsize=13)
ax.set_ylabel("Resistance (%) = 100−ASR  —  higher better", fontsize=13)
ax.set_title("Utility vs Security by Defense — GPT-OSS 120B\n(top-right = best on both axes)",
             fontsize=14, fontweight="bold")
for spn in ("top", "right"):
    ax.spines[spn].set_visible(False)
ax.grid(False)
plt.tight_layout()
p2 = OUT / "gptoss_defense_scatter.png"
fig.savefig(p2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p2}")
