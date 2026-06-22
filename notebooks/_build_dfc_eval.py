"""Builds notebooks/dfc_eval.ipynb. Run: uv run python notebooks/_build_dfc_eval.py"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# DFC vs no-defense — end-to-end ASR eval

Runs one domain **with** the admission-controlled DFC defense and **without** any defense, and reports
**ASR** (attack success rate) and clean **utility** for each.

- **ASR** = mean `security()` over the attacked runs = fraction of injections that **succeeded**.
  **Lower is better** (the defense stopped more injections).
- **Utility (clean)** = fraction of no-injection tasks the agent completed. We want DFC to **lower ASR
  without lowering utility** — a defense that blocks attacks by breaking the task is worthless.

Models: agent **Sonnet 4.6**, DFC policy-gen **Opus 4.8**, sink classifier **Haiku 4.5** (all Bedrock).
Just run the cells top-to-bottom. The run cell prints live progress and writes
`runs/eval_dfc_shopping.jsonl` as it goes; you can interrupt and re-run the scoreboard on partial results.""")

md("## 1. Setup")
code("""import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("BEDROCK_TIMEOUT", "150")
os.environ["DFC_CLASSIFIER_MODEL"] = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # Layer-2 classifier
logging.getLogger().setLevel(logging.ERROR)
for n in ("httpx", "anthropic", "boto3", "botocore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

CWD = Path.cwd()
REPO = next((p for p in [CWD, CWD.parent, CWD / "AgentDyn"] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {CWD}"
RUNS = REPO / "runs"

from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig, AgentPipeline
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger, LOGGER_STACK
print("repo:", REPO)""")

md("## 2. Config")
code("""SUITE  = "shopping"
VER    = "v1.2"
ATTACK = "important_instructions"

AGENT     = "us.anthropic.claude-sonnet-4-6"   # the task agent
DFC_MODEL = "us.anthropic.claude-opus-4-8"     # DFC fact-extraction + policy generation
# classifier model is the DFC_CLASSIFIER_MODEL env var set in setup (Haiku 4.5)

# Scope. Subset by default (~30 min). For the full suite use:  USER_TASKS = list(suite.user_tasks)
USER_TASKS = [f"user_task_{i}" for i in range(5)]
INJ_TASKS  = [f"injection_task_{i}" for i in range(9)]
MAX_WORKERS = 3            # concurrent jobs; each gets its OWN Bedrock client (never shared across threads)
PER_TASK_TIMEOUT = 240     # seconds per job

suite = get_suite(VER, SUITE)

def build(condition):
    if condition == "no_defense":
        cfg = PipelineConfig(llm=AGENT, model_id=None, defense=None,
                             system_message_name=None, system_message=None, suite_name=SUITE)
    else:  # "dfc"
        cfg = PipelineConfig(llm=AGENT, model_id=None, defense="dfc_agent_framework_integration",
                             system_message_name=None, system_message=None, suite_name=SUITE, dfc_model=DFC_MODEL)
    return AgentPipeline.from_config(cfg)

def run_job(job):
    cond, kind, ut_id, inj = job
    LOGGER_STACK.set([])               # worker thread: own logger stack; asyncio.run is legal off the main loop
    pipe = build(cond)                 # fresh pipeline + Bedrock client per job
    ut = suite.get_user_task_by_id(ut_id)
    with OutputLogger(str(RUNS), None):
        if kind == "clean":
            u, s = run_task_without_injection_tasks(suite, pipe, ut, RUNS, True, VER)
            return {"cond": cond, "kind": kind, "ut": ut_id, "inj": inj, "utility": bool(u), "security": None}
        attack = load_attack(ATTACK, suite, pipe)
        u, s = run_task_with_injection_tasks(suite, pipe, ut, attack, RUNS, True, [inj], VER)
        return {"cond": cond, "kind": kind, "ut": ut_id, "inj": inj,
                "utility": bool(list(dict(u).values())[0]), "security": bool(list(dict(s).values())[0])}

n_attack = len(USER_TASKS) * len(INJ_TASKS)
print(f"agent={AGENT}\\ndfc  ={DFC_MODEL}")
print(f"per condition: {len(USER_TASKS)} clean + {n_attack} attacked  ->  {2 * (len(USER_TASKS) + n_attack)} jobs total")""")

md("""## 3. Run — DFC vs no-defense (live progress)

Each job runs in a worker thread (no running event loop there, so the agent's `asyncio.run()` is legal)
with its **own** Bedrock client (never shared, so no cross-loop deadlock). Prints each job as it finishes.
Safe to interrupt — partial results are in `RESULTS` and the `.jsonl`, and the scoreboard works on them.""")
code("""OUT = RUNS / "eval_dfc_shopping.jsonl"

jobs = []
for cond in ("no_defense", "dfc"):
    for ut in USER_TASKS:
        jobs.append((cond, "clean", ut, None))
        for inj in INJ_TASKS:
            jobs.append((cond, "attack", ut, inj))

RESULTS = []
if OUT.exists():
    OUT.unlink()
t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(run_job, j): j for j in jobs}
    for i, fut in enumerate(as_completed(futs), 1):
        j = futs[fut]
        try:
            r = fut.result(timeout=PER_TASK_TIMEOUT); r["error"] = None
        except Exception as e:
            r = {"cond": j[0], "kind": j[1], "ut": j[2], "inj": j[3], "utility": None, "security": None, "error": repr(e)[:160]}
        RESULTS.append(r)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\\n")
        tag = f"{r['cond']}/{r['kind']}/{r['ut']}" + (f"/{r['inj']}" if r["inj"] else "")
        print(f"[{i:3}/{len(jobs)} {time.time()-t0:6.0f}s] {tag:44} util={r['utility']} sec={r['security']} {r['error'] or ''}")
print("done ->", OUT)""")

md("## 4. Scoreboard")
code("""def asr(cond):
    v = [r["security"] for r in RESULTS if r["cond"] == cond and r["kind"] == "attack" and r["security"] is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)

def util(cond):
    v = [r["utility"] for r in RESULTS if r["cond"] == cond and r["kind"] == "clean" and r["utility"] is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)

print("=" * 66)
print(f"{SUITE}  |  agent={AGENT.split('.')[-1]}  dfc-policy={DFC_MODEL.split('.')[-1]}")
print("=" * 66)
for cond in ("no_defense", "dfc"):
    a, na = asr(cond); u, nu = util(cond)
    errs = sum(1 for r in RESULTS if r["cond"] == cond and r["error"])
    a_s = f"{a*100:5.1f}%" if a is not None else "  n/a"
    u_s = f"{u*100:5.1f}%" if u is not None else "  n/a"
    print(f"  {cond:11}  ASR={a_s} (n={na})   utility(clean)={u_s} (n={nu})   errors={errs}")
print("=" * 66)
a0, _ = asr("no_defense"); a1, _ = asr("dfc")
if a0 is not None and a1 is not None:
    print(f"ASR delta (dfc - no_defense): {(a1 - a0) * 100:+.1f} pts   (negative = DFC blocked more)")
print("ASR = mean security() over attacked runs (injection succeeded). Lower = better defense.")""")

md("## 5. Where did injections succeed? (ASR detail)")
code("""# the (user_task, injection_task) pairs where the injection SUCCEEDED (security=True), per condition
for cond in ("no_defense", "dfc"):
    hits = sorted((r["ut"], r["inj"]) for r in RESULTS
                  if r["cond"] == cond and r["kind"] == "attack" and r["security"] is True)
    print(f"{cond}: {len(hits)} successful injections")
    for ut, inj in hits:
        print(f"    {ut} x {inj}")
    print()""")

md("""## Notes

- **Reading it:** DFC works if `dfc` ASR < `no_defense` ASR while `dfc` utility stays close to
  `no_defense` utility. If `no_defense` ASR is already ~0, the agent is just injection-robust on its own
  here and there's little headroom for the defense to show — switch to a weaker/more-compliant `AGENT`
  (e.g. Haiku) to create attackable headroom.
- **Scope:** widen `USER_TASKS`/`INJ_TASKS` in the config cell for the full suite (longer). Change `SUITE`
  to run another domain. `MAX_WORKERS` trades speed for Bedrock throttling risk.
- **Artifacts:** each run still writes `runs/<pipeline>/<suite>/<task>/...`; for DFC runs the
  `*_dfc/` dir holds the generated/admitted policies and the Layer-2/3 drop events — inspect with the
  `dfc_admission_sweep.ipynb` diagnostics if a result looks off.
- Generated by `notebooks/_build_dfc_eval.py` — edit the script and rebuild rather than hand-editing the
  `.ipynb`, to avoid IDE/disk desync.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).parent / "dfc_eval.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
