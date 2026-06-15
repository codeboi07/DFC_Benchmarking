"""Builds notebooks/dfc_admission_sweep.ipynb. Run: uv run python notebooks/_build_admission_sweep.py"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# DFC admission-control sweep & diagnostics

Runs the `dfc_agent_framework_integration` defense over several tasks/suites on AWS Bedrock and reports,
per task, what the **policy-generation pipeline** did — so you can see the admission-control layers working
without re-running ad-hoc scripts.

**The layers being measured**
- **Layer 1 (prompt)** — sink-selection guidance lowers the *rate* of bad policies.
- **Layer 2 (deterministic admission control)** — a neutral Haiku classifier labels each sink tool
  read-only vs effectful; policies on **read-only** sinks are **dropped** before registration. Fail-safe:
  on any classifier error, nothing is dropped (a misclassification costs utility, never security).
- **Layer 3 (enforcing)** — drops policies whose equality grounds on a *goal-phrase* fact (a loose
  description like "smart watch", not a precise target). Conservative: keeps emails/urls/paths/numbers/
  filenames/proper-noun titles, so it never drops a real egress/transfer guard.
- **Client hardening** — `BedrockStructuredLLMClient` coerces the malformed-list shape and retries on
  validation errors instead of crashing the task.

**What "leak" means here:** an *admitted* (kept) policy whose sink the classifier itself labelled
read-only — i.e. Layer 2 should have dropped it but didn't. That's the number we want at **0**.""")

md("## 1. Setup")
code("""import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

logging.getLogger().setLevel(logging.WARNING)
for n in ("httpx", "openai", "httpcore", "anthropic", "boto3", "botocore", "urllib3"):
    logging.getLogger(n).setLevel(logging.WARNING)

CWD = Path.cwd()
REPO = next((p for p in [CWD, CWD.parent, CWD / "AgentDyn"] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {CWD}"
load_dotenv(REPO / ".env")            # optional; Bedrock uses AWS creds, not OPENAI_API_KEY
os.environ.setdefault("AWS_REGION", "us-east-1")
RUNS = REPO / "runs"

def safe_name(name):
    # Bedrock model ids contain ':' (illegal in Windows paths); the benchmark sanitizes ':' '/' -> '_'
    return name.replace("/", "_").replace(":", "_")

def load_meta(p, retries=4):
    # robust JSON read: metadata.json is rewritten via truncate+write (non-atomic), so a read can land
    # mid-write; an abandoned task's leaked thread may also still be writing. Retry, then give up.
    p = Path(p)
    if not p.exists():
        return None
    for _ in range(retries):
        try:
            t = p.read_text(encoding="utf-8").strip()
            if t:
                return json.loads(t)
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(0.4)
    return None

def read_events(p):
    # parse a *_dfc/events.jsonl into a list of dicts, skipping malformed lines
    out = []
    p = Path(p)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

from agentdojo.task_suite.load_suites import get_suite
print("repo:", REPO)""")

md("## 2. Sweep config")
code("""SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
HAIKU  = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS41 = "us.anthropic.claude-opus-4-1-20250805-v1:0"

AGENT_MODEL = SONNET            # task completion
DFC_MODEL   = OPUS41            # fact extraction + policy generation
# Layer-2 sink classifier reads DFC_CLASSIFIER_MODEL (set BEFORE building any pipeline). Default Haiku.
os.environ["DFC_CLASSIFIER_MODEL"] = HAIKU

BENCHMARK_VERSION = "v1.2"
ATTACK = "important_instructions"
PER_TASK_TIMEOUT = 200          # seconds; a wedged task is abandoned and the sweep continues

# (suite, user_task, injection_task or None).  None = clean run (exercises policy-gen + admission control).
# Add injection ids to also measure blocking; clean runs are faster and still exercise Layers 1-3.
JOBS = [
    ("shopping",  "user_task_0", None),
    ("shopping",  "user_task_2", None),
    ("workspace", "user_task_0", None),
    ("workspace", "user_task_1", None),
    ("banking",   "user_task_0", None),
    ("banking",   "user_task_2", None),
]
print("agent     :", AGENT_MODEL)
print("dfc       :", DFC_MODEL)
print("classifier:", os.environ["DFC_CLASSIFIER_MODEL"])
print("jobs      :", len(JOBS))""")

md("""## 3. Run the sweep

Each task runs in its **own 1-worker thread pool**: a worker thread has no running event loop, so the
agent's `asyncio.run()` per turn is legal (Jupyter's main thread has one and would raise); and the
per-task timeout means a registration hang is **abandoned**, not allowed to freeze the kernel.""")
code("""from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig, AgentPipeline
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger, LOGGER_STACK

_suites = {}
def cached_suite(sn):
    if sn not in _suites:
        _suites[sn] = get_suite(BENCHMARK_VERSION, sn)
    return _suites[sn]

def build_pipeline(suite_name):
    cfg = PipelineConfig(llm=AGENT_MODEL, model_id=None, defense="dfc_agent_framework_integration",
                         system_message_name=None, system_message=None, suite_name=suite_name, dfc_model=DFC_MODEL)
    return AgentPipeline.from_config(cfg)

def _run(sn, ut_id, inj_id):
    LOGGER_STACK.set([])                       # worker thread gets its own logger stack
    suite = cached_suite(sn)
    pipe = build_pipeline(sn)                   # fresh Bedrock client + DFC connection per task
    ut = suite.get_user_task_by_id(ut_id)
    with OutputLogger(str(RUNS), None):
        if inj_id is None:
            u, s = run_task_without_injection_tasks(suite, pipe, ut, RUNS, True, BENCHMARK_VERSION)
            res = {"utility": u, "security": None}
        else:
            attack = load_attack(ATTACK, suite, pipe)
            u, s = run_task_with_injection_tasks(suite, pipe, ut, attack, RUNS, True, [inj_id], BENCHMARK_VERSION)
            res = {"utility": list(dict(u).values())[0], "security": list(dict(s).values())[0]}
    res["pipeline"] = safe_name(pipe.name)
    return res

_fallback_pipe = safe_name(f"{AGENT_MODEL}-dfc_agent_framework_integration")
RESULTS = []
for (sn, ut, inj) in JOBS:
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_run, sn, ut, inj)
    try:
        r = fut.result(timeout=PER_TASK_TIMEOUT); r["error"] = None
    except FuturesTimeout:
        r = {"utility": None, "security": None, "pipeline": _fallback_pipe, "error": f"ABANDONED>{PER_TASK_TIMEOUT}s"}
    except Exception as e:
        r = {"utility": None, "security": None, "pipeline": _fallback_pipe, "error": repr(e)[:160]}
    finally:
        ex.shutdown(wait=False)                # never block on a possibly-wedged worker
    r.update({"suite": sn, "ut": ut, "inj": inj, "secs": round(time.time() - t0, 1)})
    RESULTS.append(r)
    tag = f"{sn}/{ut}" + (f"/{inj}" if inj else "")
    print(f"[{r['secs']:6.1f}s] {tag:28} util={r['utility']} sec={r['security']} err={r['error']}")
print("sweep done")""")

md("""## 4. Per-task diagnostics

For each task: the classifier's read-only verdict, what **Layer 2 dropped**, the **admitted** policies,
any **leak** (admitted policy on a read-only sink), registration deletes, Layer-3 groundability warnings,
and any validation blocks.""")
code("""def artifact_dir(r):
    attack_dir = "none" if r["inj"] is None else ATTACK
    stem = "none" if r["inj"] is None else r["inj"]
    return RUNS / r["pipeline"] / r["suite"] / r["ut"] / attack_dir / f"{stem}_dfc"

def _read_only_set(ev):
    cls = [e for e in ev if e.get("event") == "sink_classification_complete"]
    return set(cls[-1].get("read_only", [])) if cls else set()

def diagnose(r):
    d = artifact_dir(r)
    md_ = load_meta(d / "metadata.json")
    ev = read_events(d / "events.jsonl")
    label = f"{r['suite']}/{r['ut']}" + (f"/{r['inj']}" if r["inj"] else "")
    print("=" * 86)
    print(f"{label}   util={r['utility']} security={r['security']}  ({r['secs']}s)")
    if r["error"]:
        print("  RUN ERROR:", r["error"])
    if md_ is None:
        print("  (no readable metadata — abandoned or mid-write)"); return
    read_only = _read_only_set(ev)
    cls_failed = any(e.get("event") == "sink_classification_failed" for e in ev)
    dropped = [e.get("sink") for e in ev if e.get("event") == "policy_dropped_read_only_sink"]
    l3_dropped = [(e.get("sink"), e.get("goal_phrase_facts")) for e in ev if e.get("event") == "policy_dropped_goal_phrase_grounding"]
    admitted = md_.get("generated_policies", [])
    admitted_sinks = [p.get("applies_to_relation") for p in admitted]
    leaks = [s for s in admitted_sinks if s in read_only]
    print(f"  classifier : {'FAILED (fail-safe: kept all)' if cls_failed else 'read_only=' + str(sorted(read_only))}")
    print(f"  Layer 2 dropped (read-only sink)      : {dropped or 'none'}")
    print(f"  Layer 3 dropped (goal-phrase grounding): {l3_dropped or 'none'}")
    print(f"  admitted ({len(admitted)}) : {admitted_sinks}")
    if leaks:
        print(f"  !!! LEAK — admitted policy on a read-only sink: {leaks}")
    print(f"  registered={len(md_.get('registered_policy_ids', []))} deleted={len(md_.get('deleted_policies', []))}")
    blocked = [e for e in md_.get("validation_events", []) if e.get("blocked")]
    if blocked:
        print(f"  validation BLOCKS ({len(blocked)}):")
        for b in blocked[:6]:
            print(f"     {b.get('event_type')} :: {b.get('policy_descriptions')}")

for r in RESULTS:
    diagnose(r)""")

md("## 5. Scoreboard")
code("""def is_leak(r):
    d = artifact_dir(r)
    md_ = load_meta(d / "metadata.json")
    if md_ is None:
        return False
    ro = _read_only_set(read_events(d / "events.jsonl"))
    return any(p.get("applies_to_relation") in ro for p in md_.get("generated_policies", []))

def _count(r, event):
    return sum(1 for e in read_events(artifact_dir(r) / "events.jsonl") if e.get("event") == event)

def n_l2(r):
    return _count(r, "policy_dropped_read_only_sink")

def n_l3(r):
    return _count(r, "policy_dropped_goal_phrase_grounding")

total = len(RESULTS)
crashed = sum(1 for r in RESULTS if r["error"])
leaks = sum(1 for r in RESULTS if is_leak(r))
l2 = sum(n_l2(r) for r in RESULTS)
l3 = sum(n_l3(r) for r in RESULTS)
util_ok = sum(1 for r in RESULTS if r["utility"] is True)

print("=" * 66)
print("SCOREBOARD")
print(f"  tasks run               : {total}")
print(f"  crashes / abandons      : {crashed}")
print(f"  benign-sink LEAKS       : {leaks}   (admitted policy on a read-only sink -> want 0)")
print(f"  Layer 2 drops (read-only): {l2}   (fired on {sum(1 for r in RESULTS if n_l2(r))} task(s))")
print(f"  Layer 3 drops (goal-phrase): {l3}   (fired on {sum(1 for r in RESULTS if n_l3(r))} task(s))")
print(f"  utility = True          : {util_ok}/{total}")
print("=" * 66)
print(f"{'task':30} {'util':6} {'sec':6} {'leak':5} {'L2':>3} {'L3':>3} error")
for r in RESULTS:
    lab = f"{r['suite']}/{r['ut']}" + (f"/{r['inj']}" if r["inj"] else "")
    print(f"{lab:30} {str(r['utility']):6} {str(r['security']):6} {str(is_leak(r)):5} {n_l2(r):>3} {n_l3(r):>3} {r['error'] or ''}")""")

md("""## Notes

- **Leak = 0 is the goal.** A leak is an *admitted* policy on a sink the classifier itself called
  read-only — Layer 2 should have dropped it. (Distinct from the generator simply not producing one,
  which shows as `dropped=[]` with no admitted read-only sink.)
- **Layer 2 is the guarantee, not Layer 1.** Generation is non-deterministic; the prompt only lowers the
  rate. Watch the `Layer 2 dropped` line to see the deterministic filter actually firing (e.g. it drops
  `ReadFileInput` on banking). It judges each tool from its **own description**, so it generalizes to
  suites/tools never tested — no hardcoded name list.
- **Fail-safe direction:** if the classifier errors, it drops nothing (`classifier: FAILED`). A
  misclassification costs utility (a kept false-positive), never security (a dropped guard).
- **Utility=False with no leak** is a separate agent-incompletion issue, not an admission-control bug.
- **Cost/latency:** Opus policy-gen is ~75-95s/task; the Haiku classifier adds one cheap call. Clean
  (no-injection) jobs are fastest and still exercise Layers 1-3. Add injection ids to a job to also
  measure live blocking.
- Override the classifier with `DFC_CLASSIFIER_MODEL` (set in the config cell before building pipelines).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).parent / "dfc_admission_sweep.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
