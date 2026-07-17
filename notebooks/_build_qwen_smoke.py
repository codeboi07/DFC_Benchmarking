"""Builds notebooks/qwen_shopping_smoke.ipynb. Run: uv run python notebooks/_build_qwen_smoke.py"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Qwen3-235B × shopping — DFC smoke test

Smoke test of the full pipeline with **Qwen3-235B as the agent** on the shopping suite, **with** and
**without** the DFC defense. Qwen is weaker / more injection-compliant than the frontier Claude models, so
the no-defense baseline should have **attackable ASR > 0** (unlike Sonnet 4.6, which was 0%), giving the
defense something to actually reduce.

**Routing (two regions, on purpose):**
- **agent** = `qwen3-235b` → LiteLLM → Bedrock **Converse** application inference profile in **us-west-2**
  (tagged `project=data-flow-control`, `billing-tag1=pr2789` for lab cost tracking).
- **DFC policy-gen** = Opus 4.8 and **classifier** = Haiku 4.5 → direct Anthropic Bedrock in **us-east-1**.

ASR = mean `security()` over attacked runs (injection succeeded); **lower with DFC = the defense works**.
This is a *smoke* scope (a few tasks); widen `USER_TASKS`/`INJ_TASKS` for a real eval.""")

md("## 1. Setup")
code("""import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# agent runs in us-west-2 via litellm; the Bedrock judges (Opus/Haiku) run in us-east-1 (per-model routing)
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("BEDROCK_TIMEOUT", "150")
os.environ["DFC_CLASSIFIER_MODEL"] = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
logging.getLogger().setLevel(logging.ERROR)
for n in ("litellm", "LiteLLM", "httpx", "anthropic", "boto3", "botocore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

CWD = Path.cwd()
REPO = next((p for p in [CWD, CWD.parent, CWD / "AgentDyn"] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {CWD}"
RUNS = REPO / "runs"

import litellm  # noqa: F401  (ensures the agent route is importable; install with `uv pip install litellm`)
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

AGENT     = "qwen3-235b"                       # LiteLLM-routed Bedrock Converse profile (us-west-2)
DFC_MODEL = "us.anthropic.claude-opus-4-8"     # DFC fact-extraction + policy generation (us-east-1)

# Smoke scope. The injections below map onto sinks DFC guards: 3=email password exfil, 4=money transfer,
# 8=password change. Widen for a real eval (full grid = 20 user x 9 injection).
USER_TASKS = ["user_task_0", "user_task_1", "user_task_2"]
INJ_TASKS  = ["injection_task_3", "injection_task_4", "injection_task_8"]
MAX_WORKERS = 2            # concurrent jobs; each gets its own clients (never shared across threads)
PER_TASK_TIMEOUT = 240

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
    LOGGER_STACK.set([])                # worker thread: own logger stack
    pipe = build(cond)                  # fresh pipeline + clients per job
    ut = suite.get_user_task_by_id(ut_id)
    with OutputLogger(str(RUNS), None):
        if kind == "clean":
            u, s = run_task_without_injection_tasks(suite, pipe, ut, RUNS, True, VER)
            return {"cond": cond, "kind": kind, "ut": ut_id, "inj": inj, "utility": bool(u), "security": None}
        attack = load_attack(ATTACK, suite, pipe)
        u, s = run_task_with_injection_tasks(suite, pipe, ut, attack, RUNS, True, [inj], VER)
        return {"cond": cond, "kind": kind, "ut": ut_id, "inj": inj,
                "utility": bool(list(dict(u).values())[0]), "security": bool(list(dict(s).values())[0])}

print(f"agent={AGENT}  dfc={DFC_MODEL}")
print(f"per condition: {len(USER_TASKS)} clean + {len(USER_TASKS)*len(INJ_TASKS)} attacked")""")

md("""## 3. Run — Qwen, DFC vs no-defense (live progress)

Each job runs in a worker thread with its own clients; safe to interrupt (results are in `RESULTS`).""")
code("""jobs = []
for cond in ("no_defense", "dfc"):
    for ut in USER_TASKS:
        jobs.append((cond, "clean", ut, None))
        for inj in INJ_TASKS:
            jobs.append((cond, "attack", ut, inj))

RESULTS = []
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
        tag = f"{r['cond']}/{r['kind']}/{r['ut']}" + (f"/{r['inj']}" if r["inj"] else "")
        print(f"[{i:3}/{len(jobs)} {time.time()-t0:6.0f}s] {tag:44} util={r['utility']} sec={r['security']} {r['error'] or ''}")
print("done")""")

md("## 4. Scoreboard")
code("""def asr(cond):
    v = [r["security"] for r in RESULTS if r["cond"] == cond and r["kind"] == "attack" and r["security"] is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)

def util(cond):
    v = [r["utility"] for r in RESULTS if r["cond"] == cond and r["kind"] == "clean" and r["utility"] is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)

print("=" * 64)
print(f"{SUITE}  |  agent=qwen3-235b  dfc-policy=opus-4-8")
print("=" * 64)
for cond in ("no_defense", "dfc"):
    a, na = asr(cond); u, nu = util(cond)
    errs = sum(1 for r in RESULTS if r["cond"] == cond and r["error"])
    a_s = f"{a*100:5.1f}%" if a is not None else "  n/a"
    u_s = f"{u*100:5.1f}%" if u is not None else "  n/a"
    print(f"  {cond:11}  ASR={a_s} (n={na})   utility(clean)={u_s} (n={nu})   errors={errs}")
print("=" * 64)
a0, _ = asr("no_defense"); a1, _ = asr("dfc")
if a0 is not None and a1 is not None:
    print(f"ASR delta (dfc - no_defense): {(a1 - a0) * 100:+.1f} pts   (negative = DFC blocked more)")
print("ASR = mean security() over attacked runs (injection succeeded). Lower = better defense.")""")

md("""## Notes

- **Smoke first.** This confirms the Qwen-agent + DFC pipeline runs end-to-end on shopping and gives a
  first ASR signal. If `no_defense` ASR is still ~0%, even Qwen resists these injections — widen
  `INJ_TASKS` or `USER_TASKS`, or check the agent is actually attempting the injected action.
- **Cost tracking:** the agent's Bedrock spend is attributed via the `dfc-qwen3-235b-2507` application
  inference profile tags (`project=data-flow-control`, `billing-tag1=pr2789`). The Opus/Haiku judges bill
  in us-east-1 separately.
- **Policies** for each DFC run are saved to `runs/<pipeline>/.../*_dfc/metadata.json` and appended to
  `runs/policy_ledger.jsonl`; browse them with `uv run python notebooks/_policy_catalog.py`.
- Generated by `notebooks/_build_qwen_smoke.py` — edit the script and rebuild rather than hand-editing the
  `.ipynb`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).parent / "qwen_shopping_smoke.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
