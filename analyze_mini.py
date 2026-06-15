"""Compare baseline vs DFC on the mini-batch (5 read-only workspace tasks x injection_task_0).

For each task prints baseline (util/sec) vs DFC (util/sec), plus from DFC metadata:
 - which tool calls DFC blocked (to spot Type-2 false positives on legit read tools)
 - whether a SendEmailInput guard was registered (would catch the injection's send)
"""

import json
from pathlib import Path

RUNS = Path("runs")
BASE = RUNS / "gpt-4o-2024-05-13" / "workspace"
DFC = RUNS / "gpt-4o-2024-05-13-dfc_agent_framework_integration" / "workspace"
ATK = "important_instructions"
INJ = "injection_task_0"
TASKS = [f"user_task_{i}" for i in range(5)]

LEGIT_READ_PREFIXES = ("get_", "search_", "read_", "current_")  # read-only tools


def load(p):
    return json.load(open(p)) if p.exists() else None


print(f"{'task':<13}{'BASE util/sec':>16}{'DFC util/sec':>16}   notes")
print("-" * 90)
n_base_u = n_base_s = n_dfc_u = n_dfc_s = 0
fp_tasks = []
for t in TASKS:
    b = load(BASE / t / ATK / f"{INJ}.json")
    d = load(DFC / t / ATK / f"{INJ}.json")
    notes = []
    bu = bs = du = ds = None
    if b:
        bu, bs = b["utility"], b["security"]
        n_base_u += bu
        n_base_s += bs
    if d:
        du, ds = d["utility"], d["security"]
        n_dfc_u += du
        n_dfc_s += ds
    # DFC diagnostics
    meta = load(DFC / t / ATK / f"{INJ}_dfc" / "metadata.json")
    blocked_tools = set()
    send_guard = False
    if meta:
        diag = meta.get("diagnostics", meta)
        for e in diag.get("validation_events", []) or []:
            if e.get("blocked") and e.get("event_type", "").startswith("tool_call:"):
                blocked_tools.add(e["event_type"].split(":", 1)[1])
        for pid in (meta.get("registered_policy_ids") or diag.get("registered_policy_ids") or []):
            if "SendEmail" in pid:
                send_guard = True
    blocked_reads = [x for x in blocked_tools if x.startswith(LEGIT_READ_PREFIXES)]
    if blocked_reads:
        notes.append(f"FP! blocked legit reads: {blocked_reads}")
        fp_tasks.append(t)
    notes.append("SendEmail guard: " + ("YES" if send_guard else "no"))
    if blocked_tools:
        notes.append(f"blocked={sorted(blocked_tools)}")
    bstr = f"{bu}/{bs}" if b else "—"
    dstr = f"{du}/{ds}" if d else "—"
    print(f"{t:<13}{bstr:>16}{dstr:>16}   {' | '.join(notes)}")

n = len(TASKS)
print("-" * 90)
print(f"BASELINE: utility {n_base_u}/{n} ({100*n_base_u/n:.0f}%)  security {n_base_s}/{n} ({100*n_base_s/n:.0f}%)")
print(f"DFC     : utility {n_dfc_u}/{n} ({100*n_dfc_u/n:.0f}%)  security {n_dfc_s}/{n} ({100*n_dfc_s/n:.0f}%)")
print(f"Type-2 false positives (DFC broke legit read tools): {len(fp_tasks)}/{n} -> {fp_tasks}")
print("\nNote: security=True means the INJECTION SUCCEEDED (attack won). Lower security is better.")
