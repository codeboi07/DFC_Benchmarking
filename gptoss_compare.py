"""Baseline vs DFC comparison for GPT-OSS 120B (workspace, important_instructions).

Agent: GPT-OSS 120B (vulnerable). DFC policy model: Opus 4.1. Compares on the pairs both arms
completed (DFC stopped at 553/560 when the AWS budget guardrail tripped).

security==True  => the INJECTION SUCCEEDED (attack won). ASR = mean(security). resistance = 100-ASR.
"""

import json
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("runs/gpt-oss-120b/workspace")
DFC = Path("runs/gpt-oss-120b-dfc_agent_framework_integration/workspace")
OUT = Path("results/dfc_comparison")
OUT.mkdir(parents=True, exist_ok=True)
SINK_TOOLS = {"send_email", "send_money", "send_channel_message", "send_direct_message"}


def collect(arm: Path):
    out = {}
    for f in glob.glob(str(arm / "user_task_*/important_instructions/injection_task_*.json")):
        if "metadata" in f:
            continue
        d = json.load(open(f))
        if d.get("attack_type") != "important_instructions" or d.get("injection_task_id") is None:
            continue
        if d.get("utility") is None or d.get("security") is None:
            continue
        ut, it = f.split("/")[-3], f.split("/")[-1][:-5]
        out[(ut, it)] = {"utility": bool(d["utility"]), "security": bool(d["security"])}
    return out


def dfc_blocked_sink(ut, it):
    mp = glob.glob(str(DFC / ut / "important_instructions" / f"{it}_dfc" / "metadata.json"))
    if not mp:
        return False
    m = json.load(open(mp[0]))
    for e in m.get("validation_events", []):
        if e.get("blocked") and str(e.get("event_type", "")).startswith("tool_call:"):
            if e["event_type"].split(":", 1)[1] in SINK_TOOLS:
                return True
    return False


base, dfc = collect(BASE), collect(DFC)
common = sorted(set(base) & set(dfc))
n = len(common)

b_u = 100 * sum(base[k]["utility"] for k in common) / n
b_s = 100 * sum(base[k]["security"] for k in common) / n
d_u = 100 * sum(dfc[k]["utility"] for k in common) / n
d_s = 100 * sum(dfc[k]["security"] for k in common) / n

# Type 1/2/3
t1 = t2 = t3 = clean = 0
for k in common:
    b, d = base[k], dfc[k]
    caught = b["security"] and not d["security"]
    missed = d["security"]
    fp = b["utility"] and not d["utility"]
    if caught:
        t1 += 1
    if missed:
        t3 += 1
    if fp:
        t2 += 1
    if not (caught or missed or fp):
        clean += 1

print("=" * 76)
print("GPT-OSS 120B — Baseline vs DFC (workspace, important_instructions)".center(76))
print("=" * 76)
print("CONVENTION: security==True => attack SUCCEEDED. ASR = mean(security==True),")
print("            lower=better. resistance = 100 - ASR. (security==False is NOT ASR.)")
print(f"baseline completed: {len(base)}/560   DFC completed: {len(dfc)}/560   compared (common): {n}")
print(f"\n{'arm':<10}{'utility%':>12}{'ASR%':>10}{'resistance%':>14}")
print(f"{'baseline':<10}{b_u:>12.1f}{b_s:>10.1f}{100-b_s:>14.1f}")
print(f"{'DFC':<10}{d_u:>12.1f}{d_s:>10.1f}{100-d_s:>14.1f}")
print(f"\nDFC effect: ASR {b_s:.1f}% -> {d_s:.1f}%  ({b_s-d_s:+.1f} pts)   utility {b_u:.1f}% -> {d_u:.1f}%  ({d_u-b_u:+.1f} pts)")
print(f"\nOutcome classification over {n} pairs:")
print(f"  Type 1  caught attack (good)   : {t1}")
print(f"  Type 2  false positive (bad)   : {t2}")
print(f"  Type 3  missed attack (bad)    : {t3}")
print(f"  Clean   (no attack to catch)   : {clean}")
print("=" * 76)

# ---------- Plot 1: utility & ASR ----------
fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
groups = ["Utility\n(higher better)", "Attack Success Rate\n(lower better)"]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w / 2, [b_u, b_s], w, label="Baseline", color="#90A4AE", edgecolor="white")
b2 = ax.bar(x + w / 2, [d_u, d_s], w, label="DFC (Opus policies)", color="#1565C0", edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100); ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel("Rate (%)", fontsize=12)
ax.set_title(f"GPT-OSS 120B: Baseline vs DFC\n(workspace, important_instructions, n={n})", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, frameon=False); ax.grid(axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(OUT / "gptoss_baseline_vs_dfc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {OUT/'gptoss_baseline_vs_dfc.png'}")

# ---------- Plot 2: Type 1/2/3 ----------
fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
bottom = 0
for label, val, c in [("Type 1: caught attack (good)", t1, "#2E7D32"),
                      ("Type 2: false positive (bad)", t2, "#F9A825"),
                      ("Type 3: missed attack (bad)", t3, "#C62828")]:
    ax.bar(["DFC"], [val], bottom=bottom, label=f"{label}  (n={val})", color=c, edgecolor="white", width=0.5)
    if val:
        ax.text(0, bottom + val / 2, str(val), ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    bottom += val
ax.set_ylabel("# (user_task x injection) pairs", fontsize=12)
ax.set_title(f"DFC Outcome Classification — GPT-OSS 120B\n({n} pairs)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(OUT / "gptoss_dfc_classification.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {OUT/'gptoss_dfc_classification.png'}")

# ---------- Plot 3: tradeoff ----------
fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
ax.scatter([b_u], [100 - b_s], s=260, color="#90A4AE", edgecolor="black", zorder=3)
ax.scatter([d_u], [100 - d_s], s=260, color="#1565C0", edgecolor="black", zorder=3)
ax.annotate("Baseline", (b_u, 100 - b_s), textcoords="offset points", xytext=(8, 6), fontweight="bold")
ax.annotate("DFC", (d_u, 100 - d_s), textcoords="offset points", xytext=(8, 6), fontweight="bold")
ax.annotate("", xy=(d_u, 100 - d_s), xytext=(b_u, 100 - b_s), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("Utility (%) — higher better", fontsize=12)
ax.set_ylabel("Attack Resistance (%) = 100-ASR — higher better", fontsize=12)
ax.set_title(f"Utility–Security Tradeoff — GPT-OSS 120B\n(DFC moves up = more secure; left = utility cost)", fontsize=12, fontweight="bold")
ax.grid(alpha=0.25, ls="--"); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(OUT / "gptoss_tradeoff.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {OUT/'gptoss_tradeoff.png'}")
print(f"\nAll outputs -> {OUT}/")
