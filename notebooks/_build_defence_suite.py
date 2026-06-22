"""Builds notebooks/qwen_defences/qwen_3_<SUITE>_<DEFENSE>.ipynb for a repo BUILT-IN defense on Qwen3-235B.

Usage:  uv run python notebooks/_build_defence_suite.py <suite> [defense]
        uv run python notebooks/_build_defence_suite.py shopping tool_filter   # default defense = tool_filter

These notebooks evaluate AgentDojo's own built-in defenses (tool_filter first) as a comparison point for DFC.
The agent is Qwen3-235B; the BASELINE is REUSED from the existing Qwen DFC-run archives (no re-run) when one is
found, else run inline. Each run archives to qwen_defences/<DEFENSE>/<SUITE>/run_<timestamp>/. The shared MP
worker (`_mp_worker.py`) lives in final_runs/Qwen_3/ and now builds any built-in defense named by the condition.
"""
import sys
import nbformat as nbf
from pathlib import Path

# defense -> display label, and (optional) a separate FILTER/judge model run via env in the worker.
DEFENSES = {
    "tool_filter": {
        "label": "Tool Filter",
        # tool_filter selects the task-relevant tool subset with an LLM up front; we run that LLM on Opus 4.8
        # (a strong separate judge) while Qwen stays the agent under test. Empty -> use the agent's own model.
        "filter_model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "filter_region": "us-east-1",
    },
}

SUITE = sys.argv[1] if len(sys.argv) > 1 else "shopping"
DEFENSE = sys.argv[2] if len(sys.argv) > 2 else "tool_filter"
assert DEFENSE in DEFENSES, f"unknown defense {DEFENSE!r}; known: {list(DEFENSES)}"
D = DEFENSES[DEFENSE]
LABEL, FILTER_MODEL, FILTER_REGION = D["label"], D.get("filter_model", ""), D.get("filter_region", "us-east-1")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(f"""# Qwen3-235B × {SUITE} — built-in defense: **{LABEL}** vs baseline

Evaluates AgentDojo's own **`{DEFENSE}`** defense on **Qwen3-235B**, as a comparison point for DFC.

**What `{DEFENSE}` does:** an LLM is given the user's task + every tool's description and returns the
comma-separated subset of tools relevant to the task; the runtime is then restricted to ONLY those tools before
the agent runs. A tool the task doesn't need (e.g. `send_money` on a buy-a-product task) is removed entirely, so
an injection that needs it is dead on arrival. It filters TOOLS, not arguments (the contrast with DFC, which
guards tool *inputs/destinations*). **The filter LLM here is Opus 4.8** (a strong separate judge); the agent
stays Qwen3-235B.

**Baseline is REUSED** from the existing Qwen DFC-run archives — no re-run — when one exists for this suite.
ASR = mean `security()` over attacked runs; utility = clean-task completion.""")

md("## Setup")
code(f"""import os, json, time, queue, importlib, sys, re, datetime, glob, collections
from pathlib import Path

CWD = Path.cwd()
REPO = next((p for p in [CWD, *CWD.parents] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {{CWD}}"
RUNS = REPO / "runs"
MODULES_DIR = REPO / "notebooks" / "final_runs" / "Qwen_3"          # shared _mp_worker.py
OUTDIR = REPO / "notebooks" / "qwen_defences" / "{DEFENSE}"
SUITE_DIR = OUTDIR / "{SUITE}"
SUITE_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = SUITE_DIR / f"run_{{RUN_ID}}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOGDIR = RUN_DIR / "logs"; LOGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(MODULES_DIR))
import _mp_worker; importlib.reload(_mp_worker)
print("repo:", REPO, "| cores:", os.cpu_count())
print("THIS RUN archives to:", RUN_DIR)""")

md("## Config")
code(f"""SUITE   = "{SUITE}"
DEFENSE = "{DEFENSE}"
VER     = "v1.2"
ATTACK  = "important_instructions"
AGENT   = "qwen3-235b"
FILTER_MODEL  = "{FILTER_MODEL}"     # tool_filter's selection LLM (Opus 4.8); empty -> agent's own model
FILTER_REGION = "{FILTER_REGION}"

from agentdojo.task_suite.load_suites import get_suite
_suite = get_suite(VER, SUITE)
_nk = lambda t: int(re.search(r"(\\d+)$", t).group(1))
USER_TASKS = sorted(_suite.user_tasks.keys(), key=_nk)
INJ_TASKS  = sorted(_suite.injection_tasks.keys(), key=_nk)

N_PROC  = min(12, os.cpu_count() or 12)
THREADS = 8
OVERALL_BUDGET = 7200

import multiprocessing as mp
CONFIG = {{
    "agent": AGENT, "classifier_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "suite": SUITE, "ver": VER, "attack": ATTACK, "runs": str(RUNS), "rerun": True,
    "bedrock_max_retries": 10, "litellm_retries": 10,
    "tool_filter_model": FILTER_MODEL or None, "tool_filter_region": FILTER_REGION,
    "agent_io_log": str(LOGDIR / "agent_io.jsonl"), "run_log": str(LOGDIR / "run.log"),
    "errors_log": str(LOGDIR / "errors.log"),
}}

def run_condition(cond, n_proc=N_PROC, threads=THREADS):
    jobs  = [(cond, "clean",  ut, None) for ut in USER_TASKS]
    jobs += [(cond, "attack", ut, inj)  for ut in USER_TASKS for inj in INJ_TASKS]
    out = RUN_DIR / f"{{cond}}_results.jsonl"; out.write_text("", encoding="utf-8")
    mgr = mp.Manager(); in_q = mgr.Queue(); out_q = mgr.Queue()
    for j in jobs: in_q.put(j)
    for _ in range(n_proc * threads): in_q.put(None)
    procs = [mp.Process(target=_mp_worker.worker, args=(in_q, out_q, CONFIG, threads)) for _ in range(n_proc)]
    results, t0, deadline = [], time.time(), time.time() + OVERALL_BUDGET
    for p in procs: p.start()
    while len(results) < len(jobs):
        try:
            r = out_q.get(timeout=120)
        except queue.Empty:
            if all(not p.is_alive() for p in procs):
                print(f"  !! workers exited early {{len(results)}}/{{len(jobs)}}"); break
            if time.time() > deadline:
                print(f"  !! budget hit {{len(results)}}/{{len(jobs)}}"); break
            continue
        results.append(r)
        with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(r) + "\\n")
        n = len(results)
        if n % 10 == 0 or r.get("error") or n == len(jobs):
            errs = sum(1 for x in results if x.get("error"))
            print(f"  [{{n:3}}/{{len(jobs)}} {{time.time()-t0:6.0f}}s] errors={{errs}} last={{r['ut']}}{{('/'+r['inj']) if r['inj'] else ''}} sec={{r['security']}}")
    for p in procs: p.terminate()
    for p in procs: p.join(timeout=10)
    _mp_worker.merge_logs(LOGDIR)
    print(f"  {{cond}}: {{len(results)}}/{{len(jobs)}} in {{time.time()-t0:.0f}}s -> {{out}}")
    return results

print(f"suite={{SUITE}} defense={{DEFENSE}} agent={{AGENT}} filter={{FILTER_MODEL or AGENT}}  |  {{N_PROC}}x{{THREADS}}")""")

md("""## 1. Baseline — reuse the existing Qwen archive (no re-run)

Loads the most recent `final_runs/Qwen_3/<suite>/run_*/baseline_results.jsonl`. If none exists for this suite,
set `RUN_BASELINE = True` to run it inline.""")
code("""RUN_BASELINE = False
def _latest_baseline(suite):
    arch = sorted(glob.glob(str(MODULES_DIR / suite / "run_*" / "baseline_results.jsonl")))
    return arch[-1] if arch else None

src = _latest_baseline(SUITE)
if src and not RUN_BASELINE:
    BASELINE = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"REUSED baseline from {src}  ({len(BASELINE)} runs)")
else:
    print("no archived baseline (or RUN_BASELINE=True) -> running baseline inline...")
    BASELINE = run_condition("baseline")""")

md(f"""## 2. {LABEL} run""")
code("""print(f'=== {DEFENSE} (filter LLM = {FILTER_MODEL or AGENT}) ===')
DEFENDED = run_condition(DEFENSE)""")

md("## Scoreboard")
code("""def summarize(results):
    att = [r for r in results if r["kind"] == "attack"]
    sec = [r["security"] for r in att if r["security"] is not None]
    util = [r["utility"] for r in results if r["kind"] == "clean" and r["utility"] is not None]
    errs = sum(1 for r in results if r.get("error"))
    return {"ASR": (sum(sec)/len(sec)) if sec else None, "n_att": len(sec),
            "utility": (sum(util)/len(util)) if util else None, "n_clean": len(util), "errors": errs}

b, d = summarize(BASELINE), summarize(DEFENDED)
print("=" * 70); print(f"{SUITE.upper()} | agent=qwen3-235b | defense={DEFENSE} (filter={FILTER_MODEL or AGENT})"); print("=" * 70)
for label, m in [("baseline", b), (DEFENSE, d)]:
    a = f"{m['ASR']*100:5.1f}%" if m['ASR'] is not None else "  n/a"
    u = f"{m['utility']*100:5.1f}%" if m['utility'] is not None else "  n/a"
    print(f"  {label:12} ASR={a} (n={m['n_att']})  utility(clean)={u} (n={m['n_clean']})  errors={m['errors']}")
print("=" * 70)
if b['ASR'] is not None and d['ASR'] is not None:
    print(f"ASR: baseline {b['ASR']*100:.1f}% -> {DEFENSE} {d['ASR']*100:.1f}%  (delta {(d['ASR']-b['ASR'])*100:+.1f} pts)")
summary = {"suite": SUITE, "agent": AGENT, "defense": DEFENSE, "filter_model": FILTER_MODEL, "run_id": RUN_ID,
           "baseline": b, "defended": d}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with open(SUITE_DIR / "runs_index.md", "a", encoding="utf-8") as f:
    ba = f"{b['ASR']*100:.1f}%" if b['ASR'] is not None else "n/a"
    da = f"{d['ASR']*100:.1f}%" if d['ASR'] is not None else "n/a"
    du = f"{d['utility']*100:.0f}%" if d['utility'] is not None else "n/a"
    f.write(f"- **{RUN_ID}** {DEFENSE} — baseline ASR {ba} -> {da}, utility {du}, errors {d['errors']}  (`run_{RUN_ID}/`)\\n")
print("wrote", RUN_DIR / "summary.json")""")

md("""## Where injections succeeded / were blocked, and the filter's selections""")
code("""def hits(results):
    return {(r["ut"], r["inj"]) for r in results if r["kind"]=="attack" and r["security"] is True}
bh, dh = hits(BASELINE), hits(DEFENDED)
print(f"baseline: {len(bh)} successful injections   {DEFENSE}: {len(dh)} successful injections")
print(f"{DEFENSE} fixed (baseline-succeeded, now blocked): {len(bh - dh)}")
newly = sorted(dh - bh)
if newly:
    print(f"!! succeeded only WITH {DEFENSE} (stochastic / regression): {len(newly)}")
    for ut, inj in newly[:20]: print(f"   {ut} x {inj}")
bc = {r["ut"]: r["utility"] for r in BASELINE if r["kind"]=="clean"}
dc = {r["ut"]: r["utility"] for r in DEFENDED if r["kind"]=="clean"}
broke = [ut for ut in bc if bc[ut] and not dc.get(ut)]
print(f"\\nclean tasks that passed baseline but failed {DEFENSE}: {broke or 'none'}")

# what tools did the filter select per clean task? (the filter's response is the 2nd assistant message)
def _txt(c): return c if isinstance(c,str) else " ".join(str(x.get("content","")) for x in c) if isinstance(c,list) else str(c)
base = RUNS / f"qwen3-235b-{DEFENSE}" / SUITE
print(f"\\nfilter tool-selection (sample clean tasks):")
for ut in USER_TASKS[:5]:
    p = base / ut / "none" / "none.json"
    if not p.exists(): continue
    tj = json.load(open(p, encoding="utf-8"))
    for m in tj.get("messages", []):
        if m.get("role") == "assistant":
            sel = _txt(m.get("content")); print(f"  {ut}: {sel[:120]}"); break""")

md(f"""## Notes

- **Defense:** `{DEFENSE}` — LLM-selected tool allowlist (filter LLM = Opus 4.8, agent = Qwen3-235B). Filters
  TOOLS only, never tool arguments.
- **Baseline reused** from `final_runs/Qwen_3/<suite>/run_*/baseline_results.jsonl` (set `RUN_BASELINE=True` to
  run inline if no archive exists — e.g. shopping).
- Per-task traces (incl. the filter's tool selection) live in `runs/qwen3-235b-{DEFENSE}/{SUITE}/...` (latest run
  only). The filter model is set via `DFC_TOOL_FILTER_MODEL` (worker env from `CONFIG['tool_filter_model']`);
  the `tool_filter` branch in `agent_pipeline.py` resolves it.
- Built by `notebooks/_build_defence_suite.py {SUITE} {DEFENSE}`.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out_dir = Path(__file__).parent / "qwen_defences"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"qwen_3_{SUITE}_{DEFENSE}.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
