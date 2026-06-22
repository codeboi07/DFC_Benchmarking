"""Consolidate every generated DFC policy into one catalog you can look back at.

Reads both sources: the central runs/policy_ledger.jsonl (written going forward) AND every
runs/**/<*_dfc>/metadata.json (so it covers all past runs too). Dedupes by (suite, sink, normalized PGN
skeleton), and writes runs/policy_catalog.md + runs/policy_catalog.json.

Run: uv run python notebooks/_policy_catalog.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

CWD = Path.cwd()
REPO = next((p for p in [CWD, CWD.parent, CWD / "AgentDyn"] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {CWD}"
RUNS = REPO / "runs"


def skeleton(pgn: str) -> str:
    # normalize away task-specific fact keys and literal values so structurally-identical policies collapse
    s = pgn or ""
    s = re.sub(r"PreambleData\.\w+", "PreambleData.<FACT>", s)
    s = re.sub(r"'[^']*'", "'<VAL>'", s)
    return re.sub(r"\s+", " ", s).strip()


# catalog[suite][sink] -> { skeleton: {count, example_pgn, example_desc, tasks:set} }
catalog: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
total = 0


def add(suite, task, sink, pgn, desc):
    global total
    if not pgn or not sink:
        return
    total += 1
    sk = skeleton(pgn)
    bucket = catalog[suite or "?"][sink]
    entry = bucket.setdefault(sk, {"count": 0, "example_pgn": pgn.strip(), "example_desc": desc or "", "tasks": set()})
    entry["count"] += 1
    if task:
        entry["tasks"].add(task)


# 1) the forward ledger
ledger = RUNS / "policy_ledger.jsonl"
if ledger.exists():
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for p in rec.get("policies", []):
            add(rec.get("suite"), rec.get("task_id"), p.get("sink"), p.get("pgn"), p.get("description"))

# 2) every per-task metadata.json (covers past runs retroactively). suite is the dir under the pipeline.
for md in RUNS.glob("*/*/*/*/*_dfc/metadata.json"):
    try:
        d = json.loads(md.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue
    parts = md.parts
    # runs/<pipeline>/<suite>/<user_task>/<attack>/<inj>_dfc/metadata.json
    suite = parts[-5] if len(parts) >= 5 else None
    task = parts[-4] if len(parts) >= 4 else None
    for p in d.get("generated_policies", []):
        add(suite, task, p.get("applies_to_relation"), p.get("pgn"), p.get("description"))

# ---- write outputs ----
lines = ["# DFC policy catalog", "",
         f"Distinct policy shapes across all runs (deduped by suite + sink + structure). "
         f"Total policy instances seen: **{total}**.", ""]
catalog_json: dict = {}
for suite in sorted(catalog):
    lines.append(f"## {suite}")
    catalog_json[suite] = {}
    for sink in sorted(catalog[suite]):
        shapes = catalog[suite][sink]
        lines.append(f"\n### {sink}  ({len(shapes)} distinct shape(s))")
        catalog_json[suite][sink] = []
        for sk, e in sorted(shapes.items(), key=lambda kv: -kv[1]["count"]):
            lines.append(f"\n- seen **{e['count']}x** across {len(e['tasks'])} task(s) — {e['example_desc'][:90]}")
            lines.append("  ```")
            for ln in e["example_pgn"].splitlines():
                lines.append("  " + ln)
            lines.append("  ```")
            catalog_json[suite][sink].append({
                "count": e["count"], "tasks": sorted(e["tasks"]),
                "example_pgn": e["example_pgn"], "description": e["example_desc"],
            })

(RUNS / "policy_catalog.md").write_text("\n".join(lines), encoding="utf-8")
(RUNS / "policy_catalog.json").write_text(json.dumps(catalog_json, indent=2), encoding="utf-8")

# ---- console summary ----
print(f"policy instances seen: {total}")
print(f"distinct (suite, sink, shape) entries: {sum(len(s) for su in catalog.values() for s in su.values())}")
for suite in sorted(catalog):
    sinks = catalog[suite]
    print(f"\n{suite}:")
    for sink in sorted(sinks, key=lambda s: -sum(e['count'] for e in sinks[s].values())):
        n = sum(e["count"] for e in sinks[sink].values())
        print(f"  {sink:38} {len(sinks[sink])} shape(s)  ({n} instances)")
print(f"\nwrote {RUNS/'policy_catalog.md'}  and  {RUNS/'policy_catalog.json'}")
