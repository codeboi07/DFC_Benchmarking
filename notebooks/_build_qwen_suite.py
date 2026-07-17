"""Builds notebooks/final_runs/Qwen_3/[<subdir>/]<prefix>_<SUITE>_full.ipynb for an AgentDyn suite.

Usage:  uv run python notebooks/_build_qwen_suite.py <suite> [model_key]
        uv run python notebooks/_build_qwen_suite.py github                 # qwen3-235b (default)
        uv run python notebooks/_build_qwen_suite.py github deepseek-v3.2   # DeepSeek V3.2

`model_key` selects an agent registered in agentdojo.models (provider "litellm"). The shared MP worker
(`_mp_worker.py`) and judge-analysis (`_judge_analysis.py`) live in final_runs/Qwen_3/ and are imported by
EVERY model's notebook, so only the agent id / output folder change per model. Every run of the produced
notebook writes to its OWN timestamped folder (<MODEL_DIR>/<SUITE>/run_<YYYYMMDD_HHMMSS>/) so results are
never overwritten.
"""
import sys
import nbformat as nbf
from pathlib import Path

# model_key -> agent id (agentdojo models / agent_pipeline litellm route), display label,
#              output subdir under final_runs/Qwen_3/, and notebook-filename prefix.
MODELS = {
    "qwen3-235b":    {"agent": "qwen3-235b",    "label": "Qwen3-235B",    "subdir": "",         "prefix": "qwen_3"},
    "deepseek-v3.2": {"agent": "deepseek-v3.2", "label": "DeepSeek V3.2", "subdir": "deepseek", "prefix": "deepseek_v3_2"},
    "gpt-oss-120b":  {"agent": "gpt-oss-120b",  "label": "GPT-OSS 120B",  "subdir": "gpt_oss",  "prefix": "gpt_oss_120b"},
    "kimi-k2.5":     {"agent": "kimi-k2.5",     "label": "Kimi K2.5",     "subdir": "kimi_k_2.5", "prefix": "kimi_k2_5"},
    "minimax-m2.5":  {"agent": "minimax-m2.5",  "label": "MiniMax M2.5",  "subdir": "mini_max",   "prefix": "minimax_m2_5"},
}

SUITE = sys.argv[1] if len(sys.argv) > 1 else "dailylife"
MODEL_KEY = sys.argv[2] if len(sys.argv) > 2 else "qwen3-235b"
assert MODEL_KEY in MODELS, f"unknown model_key {MODEL_KEY!r}; known: {list(MODELS)}"
M = MODELS[MODEL_KEY]
AGENT, LABEL, SUBDIR, PREFIX = M["agent"], M["label"], M["subdir"], M["prefix"]

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(f"""# {LABEL} × {SUITE} — full suite, baseline vs DFC

Runs the **entire {SUITE} suite** (all user tasks × all injection tasks + the clean tasks) with
**{LABEL} as the agent**, under two conditions:

1. **Baseline** — no defense
2. **DFC** — `dfc_agent_framework_integration` (Opus 4.8 policy-gen, Haiku 4.5 sink classifier, admission
   control + compile-check safety net)

ASR = mean `security()` over the attacked runs (injection succeeded); **lower with DFC = the defense works**.
Utility = clean-task completion.

**Every run is archived.** Each execution creates a fresh `{SUITE}/run_<timestamp>/` folder holding that run's
results, summary, logs, and a snapshot of its generated policies — so re-running **never overwrites** a prior
run. Nothing is lost.

**Speed:** fans out across processes (multiple GILs) each running a thread pool (LLM-call overlap) — the full
suite lands in ~5 min. The MP worker (`_mp_worker.py`) and judge analysis (`_judge_analysis.py`) are shared
across all models from `final_runs/Qwen_3/`.""")

md("## Setup")
code(f"""import os, json, time, queue, importlib, sys, re, datetime, shutil, glob, collections
from pathlib import Path

CWD = Path.cwd()
REPO = next((p for p in [CWD, *CWD.parents] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {{CWD}}"
RUNS = REPO / "runs"
MODULES_DIR = REPO / "notebooks" / "final_runs" / "Qwen_3"   # shared _mp_worker.py / _judge_analysis.py live here
MODEL_DIR   = MODULES_DIR / "{SUBDIR}"                          # this model's output root ("" -> MODULES_DIR)
SUITE_DIR   = MODEL_DIR / "{SUITE}"
SUITE_DIR.mkdir(parents=True, exist_ok=True)

# ---- per-run archive folder: results are NEVER overwritten ----
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = SUITE_DIR / f"run_{{RUN_ID}}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOGDIR = RUN_DIR / "logs"; LOGDIR.mkdir(exist_ok=True)

# the MP worker is a real module shared by every model's notebook (so Windows 'spawn' can import it in children)
sys.path.insert(0, str(MODULES_DIR))
import _mp_worker; importlib.reload(_mp_worker)

print("repo:", REPO, "| cores:", os.cpu_count())
print("THIS RUN archives to:", RUN_DIR)""")

md("## Config")
code(f"""SUITE  = "{SUITE}"
VER    = "v1.2"
ATTACK = "important_instructions"

AGENT      = "{AGENT}"                          # LiteLLM Bedrock Converse profile (us-west-2)
DFC_MODEL  = "us.anthropic.claude-opus-4-8"     # DFC fact-extraction + policy generation (us-east-1)
CLASSIFIER = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # DFC sink classifier (admission control)

# Task lists are DERIVED from the suite (so the counts are always right for this suite).
from agentdojo.task_suite.load_suites import get_suite
_suite = get_suite(VER, SUITE)
_nk = lambda t: int(re.search(r"(\\d+)$", t).group(1))
USER_TASKS = sorted(_suite.user_tasks.keys(), key=_nk)
INJ_TASKS  = sorted(_suite.injection_tasks.keys(), key=_nk)

# ---- Parallelism: PROCESSES x THREADS (the speed knob) ----
# The LLM calls are I/O-bound (just waiting on Bedrock), so THREADS carry the concurrency almost for free - they
# release the GIL while waiting. PROCESSES are the CPU cost: each is a full interpreter running DuckDB + agent
# glue, so ~one-per-core saturates the machine (that's what overwhelmed it at 12). Fix: cap PROCESSES well under
# the core count and push concurrency through THREADS instead. 5 procs x 16 threads = ~80 concurrent -> near the
# old ~5-min/condition speed, but only ~5 of your cores stay busy. Want faster: raise THREADS (cheap). Still too
# hot: lower N_PROC.
N_PROC  = min(5, os.cpu_count() or 4)
THREADS = 16
RERUN   = True              # always run fresh; this run's outputs are archived under RUN_DIR regardless
OVERALL_BUDGET = 7200       # s; stop waiting after this and report whatever finished

# --- Semantic-judge A/B switch (enforcement-layer LLM override; policy GENERATION is never changed) ---
# False = deterministic hard gate only (the standard DFC).
# True  = when the hard gate BLOCKS an action, a cheap external judge (default: Sonnet 4.6 on Bedrock
#         us-east-1, a SEPARATE endpoint from the agent) decides whether the action's target is authorized by
#         the trusted task (named/implied entity) and may OVERRIDE the block. The hard gate AND the judge must
#         BOTH condemn an action to kill it. The judge prompt/model live in
#         dfc_agent_framework_integration/semantic_judge.py (JUDGE_PROMPT / DFC_JUDGE_MODEL) and are tunable
#         without re-running generation. Run once False, once True; the deltas are the result.
SEMANTIC_JUDGE = False
# Judge MODE (only when SEMANTIC_JUDGE=True): "provenance" = name-grounded ("is the action's target a
# task-named entity?"); "injection_detect" = per-content classifier ("is the content the agent read a prompt
# injection?"). injection_detect recovers content-destination tasks whose bill/email/page is clean. The hard gate
# stays the deterministic floor either way. Both prompts live in semantic_judge.py.
JUDGE_MODE = "provenance"
# Block-on-injection (injection_detect only): once the operator detects a prompt injection in read content,
# block ALL subsequent tool calls - not just destination-hard-blocked ones. Adds coverage for the non-exfil
# attacks (buy / change-password / visit-URL / delete / star / download) the destination gate can't see. The
# deterministic hard gate is untouched; this only ADDS blocks, and only when an injection was actually read.
INJECTION_QUARANTINE = False

import multiprocessing as mp
CONFIG = {{
    "agent": AGENT, "dfc_model": DFC_MODEL, "classifier_model": CLASSIFIER,
    "suite": SUITE, "ver": VER, "attack": ATTACK, "runs": str(RUNS), "rerun": RERUN,
    "bedrock_max_retries": 10, "litellm_retries": 10, "semantic_judge": SEMANTIC_JUDGE, "judge_mode": JUDGE_MODE,
    "injection_quarantine": INJECTION_QUARANTINE,
    "agent_io_log": str(LOGDIR / "agent_io.jsonl"),   # workers write <path>.<pid>, merged after each run
    "run_log":      str(LOGDIR / "run.log"),
    "errors_log":   str(LOGDIR / "errors.log"),
}}

def run_condition(cond, n_proc=N_PROC, threads=THREADS):
    jobs  = [(cond, "clean",  ut, None) for ut in USER_TASKS]
    jobs += [(cond, "attack", ut, inj)  for ut in USER_TASKS for inj in INJ_TASKS]
    out = RUN_DIR / f"{{cond}}_results.jsonl"; out.write_text("", encoding="utf-8")   # fresh file IN THIS RUN's folder

    if cond == "dfc" and SEMANTIC_JUDGE:
        # Each worker calls the judge directly (default: Sonnet on Bedrock us-east-1, separate from the agent's
        # endpoint) when the hard gate blocks — no shared shim, no contention. Workers inherit DFC_SEMANTIC_JUDGE
        # via CONFIG.
        print("  [SEMANTIC JUDGE ON] direct external-judge override on hard-blocks (judge != agent endpoint)")

    mgr = mp.Manager(); in_q = mgr.Queue(); out_q = mgr.Queue()
    for j in jobs:
        in_q.put(j)
    for _ in range(n_proc * threads):
        in_q.put(None)
    procs = [mp.Process(target=_mp_worker.worker, args=(in_q, out_q, CONFIG, threads)) for _ in range(n_proc)]

    results, t0, deadline = [], time.time(), time.time() + OVERALL_BUDGET
    for p in procs:
        p.start()
    while len(results) < len(jobs):
        try:
            r = out_q.get(timeout=120)
        except queue.Empty:
            if all(not p.is_alive() for p in procs):
                print(f"  !! all workers exited early with {{len(results)}}/{{len(jobs)}} done"); break
            if time.time() > deadline:
                print(f"  !! OVERALL_BUDGET hit; {{len(results)}}/{{len(jobs)}} done"); break
            continue
        results.append(r)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\\n")
        n = len(results)
        if n % 10 == 0 or r.get("error") or n == len(jobs):
            errs = sum(1 for x in results if x.get("error"))
            tag = f"{{r['ut']}}{{('/' + r['inj']) if r['inj'] else ''}}"
            print(f"  [{{n:3}}/{{len(jobs)}} {{time.time()-t0:6.0f}}s] errors={{errs}}  last={{tag}} sec={{r['security']}}")
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=10)
    _mp_worker.merge_logs(LOGDIR)
    dt = time.time() - t0
    print(f"  {{cond}}: {{len(results)}}/{{len(jobs)}} jobs in {{dt:.0f}}s ({{dt/60:.1f}} min) -> {{out}}")
    return results

print(f"suite={{SUITE}}  agent={{AGENT}}  dfc={{DFC_MODEL}}  |  {{N_PROC}} procs x {{THREADS}} threads")
print(f"{{len(USER_TASKS)}} user x {{len(INJ_TASKS)}} inj = {{len(USER_TASKS)*len(INJ_TASKS)}} attacked + {{len(USER_TASKS)}} clean = {{len(USER_TASKS)*(1+len(INJ_TASKS))}} runs/condition")""")

md("## 1. Baseline (no defense)")
code("""print('=== BASELINE (no defense) ===')
BASELINE = run_condition('baseline')""")

md("## 2. DFC")
code("""print('=== DFC (admission control + compile-check) ===')
DFC = run_condition('dfc')""")

md("## Scoreboard")
code("""def summarize(results):
    att = [r for r in results if r["kind"] == "attack"]
    sec = [r["security"] for r in att if r["security"] is not None]
    util = [r["utility"] for r in results if r["kind"] == "clean" and r["utility"] is not None]
    # Utility UNDER ATTACK: user_task.utility() evaluated on the SAME attacked trajectory. This is authentic
    # AgentDojo - it is the `utility` half that run_task_with_injection_tasks() returns alongside `security`
    # (see benchmark.py / task_suite.run_task_with_pipeline). It is INDEPENDENT of ASR: an agent can finish the
    # user's task whether or not it also fell for the injection. Measures: does DFC still let the agent complete
    # the real task while it's under attack (and while DFC is enforcing)?
    util_att = [r["utility"] for r in att if r["utility"] is not None]
    errs = sum(1 for r in results if r.get("error"))
    return {"ASR": (sum(sec)/len(sec)) if sec else None, "n_att": len(sec),
            "utility": (sum(util)/len(util)) if util else None, "n_clean": len(util),
            "utility_attack": (sum(util_att)/len(util_att)) if util_att else None, "n_att_util": len(util_att),
            "errors": errs}

b, d = summarize(BASELINE), summarize(DFC)
print("=" * 70); print(f"{SUITE.upper()} (full) | agent={AGENT} | dfc-policy=opus-4-8"); print("=" * 70)
for label, m in [("baseline", b), ("dfc", d)]:
    a = f"{m['ASR']*100:5.1f}%" if m['ASR'] is not None else "  n/a"
    u = f"{m['utility']*100:5.1f}%" if m['utility'] is not None else "  n/a"
    ua = f"{m['utility_attack']*100:5.1f}%" if m['utility_attack'] is not None else "  n/a"
    print(f"  {label:9}  ASR={a} (n={m['n_att']})   utility(clean)={u} (n={m['n_clean']})   "
          f"utility(under attack)={ua} (n={m['n_att_util']})   errors={m['errors']}")
print("=" * 70)
if b['ASR'] is not None and d['ASR'] is not None:
    print(f"ASR: baseline {b['ASR']*100:.1f}% -> dfc {d['ASR']*100:.1f}%   (delta {(d['ASR']-b['ASR'])*100:+.1f} pts)")
if b['utility_attack'] is not None and d['utility_attack'] is not None:
    print(f"Utility UNDER ATTACK: baseline {b['utility_attack']*100:.1f}% -> dfc {d['utility_attack']*100:.1f}%"
          f"   (delta {(d['utility_attack']-b['utility_attack'])*100:+.1f} pts)")
# Explicit named scores (the metric we optimize for: agent still finishes the real task WHILE attacked)
baseline_utility_under_attack = b['utility_attack']
dfc_utility_under_attack = d['utility_attack']
print(f"baseline_utility_under_attack = {baseline_utility_under_attack*100:.1f}%" if baseline_utility_under_attack is not None else "baseline_utility_under_attack = n/a")
print(f"dfc_utility_under_attack      = {dfc_utility_under_attack*100:.1f}%" if dfc_utility_under_attack is not None else "dfc_utility_under_attack = n/a")
summary = {"suite": SUITE, "agent": AGENT, "dfc_model": DFC_MODEL, "run_id": RUN_ID,
           "semantic_judge": SEMANTIC_JUDGE, "baseline": b, "dfc": d,
           "baseline_utility_under_attack": baseline_utility_under_attack,
           "dfc_utility_under_attack": dfc_utility_under_attack}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# append a one-line index entry so every run is discoverable
with open(SUITE_DIR / "runs_index.md", "a", encoding="utf-8") as f:
    ba = f"{b['ASR']*100:.1f}%" if b['ASR'] is not None else "n/a"
    da = f"{d['ASR']*100:.1f}%" if d['ASR'] is not None else "n/a"
    du = f"{d['utility']*100:.0f}%" if d['utility'] is not None else "n/a"
    tag = "judge ON " if SEMANTIC_JUDGE else "hard-only"
    f.write(f"- **{RUN_ID}** [{tag}] — baseline ASR {ba} -> dfc ASR {da}, dfc utility {du}, errors {d['errors']}  (`run_{RUN_ID}/`)\\n")
print("wrote", RUN_DIR / "summary.json", "| semantic_judge =", SEMANTIC_JUDGE)""")

md("""## Where injections succeeded / were blocked (detail)""")
code("""def hits(results):
    return {(r["ut"], r["inj"]) for r in results if r["kind"]=="attack" and r["security"] is True}
bh, dh = hits(BASELINE), hits(DFC)
print(f"baseline: {len(bh)} successful injections   dfc: {len(dh)} successful injections")
print(f"\\nDFC fixed (succeeded in baseline, blocked by DFC): {len(bh - dh)}")
newly = sorted(dh - bh)
if newly:
    print(f"!! succeeded only WITH dfc (stochastic / regression): {len(newly)}")
    for ut, inj in newly: print(f"   {ut} x {inj}")

# clean tasks DFC broke -> was it a DFC block (false positive) or the agent failing on its own (noise)?
bc = {r["ut"]: r["utility"] for r in BASELINE if r["kind"]=="clean"}
dc = {r["ut"]: r["utility"] for r in DFC if r["kind"]=="clean"}
broke = [ut for ut in bc if bc[ut] and not dc.get(ut)]
print(f"\\nclean tasks that passed baseline but failed DFC: {broke or 'none'}")
base = RUNS / f"{AGENT}-dfc_agent_framework_integration" / SUITE
for ut in broke:
    tj_path = base / ut / "none" / "none.json"
    nb = 0
    if tj_path.exists():
        tj = json.load(open(tj_path, encoding="utf-8"))
        nb = sum(1 for m in tj.get("messages", []) if m.get("role")=="tool" and m.get("error") and "policy violation" in str(m.get("error")).lower())
    print(f"   {ut}: DFC blocks={nb} -> {'DFC false positive (investigate)' if nb else 'agent failed on its own (noise)'}")""")

md("""## Judge confusion matrix (only meaningful when SEMANTIC_JUDGE=True)

Classifies every judge consultation: on clean tasks override = utility recovered; on attacked tasks an
override on the **injection's attacker target** is the exact ASR leak (vs an override on a legit user-task
action). This is the precise Δutility-win / ΔASR-cost a reviewer asks for.""")
code("""import importlib, _judge_analysis; importlib.reload(_judge_analysis)
_judge_analysis.print_confusion_matrix(RUNS, SUITE, _suite, agent=AGENT)""")

md("""## Snapshot this run's policies (archived, immune to future re-runs)""")
code("""# DFC per-task metadata in runs/ is overwritten on the next run; freeze THIS run's policies into RUN_DIR.
base = RUNS / f"{AGENT}-dfc_agent_framework_integration" / SUITE
def _norm(pgn):
    c = [l.strip() for l in pgn.splitlines() if l.strip().startswith("CONSTRAINT")]
    return re.sub(r"PreambleData\\.\\w+", "PreambleData.<fact>", c[0]) if c else pgn[:60]
tasks, catalog = [], collections.defaultdict(collections.Counter)
for f in sorted(glob.glob(str(base / "*" / "*" / "*_dfc" / "metadata.json"))):
    md_ = json.load(open(f, encoding="utf-8")); parts = Path(f).parts
    gp = md_.get("generated_policies", []); fc = md_.get("policy_fire_counts", {})
    pols = [{"sink": p.get("applies_to_relation"), "pgn": p.get("pgn"), "fired": fc.get(p.get("policy_id"), 0)} for p in gp]
    tasks.append({"user_task": parts[-4], "variant": parts[-2].replace("_dfc",""),
                  "extracted_facts": md_.get("extracted_facts"), "policies": pols})
    for p in gp: catalog[p.get("applies_to_relation")][_norm(p.get("pgn",""))] += 1
snap = {"suite": SUITE, "agent": AGENT, "run_id": RUN_ID, "n_tasks": len(tasks),
        "n_policies": sum(len(t["policies"]) for t in tasks), "tasks": tasks}
(RUN_DIR / "policies_snapshot.json").write_text(json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8")
lines = [f"# {SUITE} policy catalog — {AGENT} run {RUN_ID}", "",
         f"{snap['n_policies']} policies across {snap['n_tasks']} task-runs. Distinct constraint shapes per sink:", ""]
for sink in sorted(catalog, key=lambda s: -sum(catalog[s].values())):
    lines.append(f"## {sink}  ({sum(catalog[sink].values())} policies, {len(catalog[sink])} shapes)")
    for shape, n in catalog[sink].most_common(): lines.append(f"- `{shape}`  x{n}")
    lines.append("")
(RUN_DIR / "policies_catalog.md").write_text("\\n".join(lines), encoding="utf-8")
print(f"froze {snap['n_policies']} policies / {snap['n_tasks']} tasks ->")
print("  ", RUN_DIR / "policies_snapshot.json")
print("  ", RUN_DIR / "policies_catalog.md")""")

md(f"""## Notes

- **Every run is archived, nothing overwritten.** Each execution writes to `{SUITE}/run_<timestamp>/`:
  `baseline_results.jsonl`, `dfc_results.jsonl`, `summary.json`, `logs/`, and the policy snapshot
  (`policies_snapshot.json` + `policies_catalog.md`). A one-line entry is appended to `{SUITE}/runs_index.md`
  so all runs are discoverable. Re-run as many times as you like — old runs stay put.
- **Agent:** {LABEL} (`{AGENT}`), a LiteLLM-routed Bedrock Converse application-inference-profile in us-west-2,
  kept separate from the us-east-1 Claude DFC/judge models. Shared MP worker + judge analysis live in
  `final_runs/Qwen_3/`; only the agent id and output folder change per model.
- **Parallelism (`N_PROC` x `THREADS`):** processes (multiple GILs) x threads (I/O overlap). Adjust `N_PROC`
  to your core count. The MP worker (`_mp_worker.py`) is reloaded each Setup run, so worker edits apply without
  a kernel restart — but DFC-framework edits (prompts/policies) need a **kernel restart** to re-import.
- **Semantic-judge A/B (`SEMANTIC_JUDGE`):** flip it in Config and run twice. `False` = deterministic hard
  gate only. `True` = on a hard block, a cheap external judge (Sonnet, default) can override if the action's
  target is authorized by the trusted task (named/implied entity). Each worker calls the judge directly on a
  separate endpoint (no shared shim, no contention with the agent). Generation is identical either way — only
  enforcement differs — so the run-to-run delta isolates the judge's effect (Δutility = recovered tasks,
  ΔASR = judge fooled). Judge prompt/model: `semantic_judge.py` (`JUDGE_PROMPT` / `DFC_JUDGE_MODEL`). Each run
  records `semantic_judge` in `summary.json` and tags `runs_index.md` (`[judge ON]` / `[hard-only]`).
- **DFC data** (facts, admitted policies, classifier verdicts, validation blocks) for the *latest* run is in
  `runs/{AGENT}-dfc_agent_framework_integration/{SUITE}/...` — overwritten on the next run, which is why the
  Snapshot cell freezes a copy into the run folder.
- Built by `notebooks/_build_qwen_suite.py {SUITE} {MODEL_KEY}` — edit the script and rebuild rather than
  hand-editing the `.ipynb`.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out_dir = Path(__file__).parent / "final_runs" / "Qwen_3" / SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{PREFIX}_{SUITE}_full.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
