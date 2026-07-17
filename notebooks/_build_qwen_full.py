"""Builds notebooks/final_runs/Qwen_3/qwen3_shopping_full.ipynb. Run: uv run python notebooks/_build_qwen_full.py"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Qwen3-235B × shopping — full suite, baseline vs DFC

Runs the **entire shopping suite** (20 user tasks × 9 injection tasks = 180 attacked pairs + 20 clean) with
**Qwen3-235B as the agent**, under two conditions:

1. **Baseline** — no defense
2. **DFC** — `dfc_agent_framework_integration` (Opus 4.8 policy-gen, Haiku 4.5 sink classifier, admission
   control + feedback loop)

ASR = mean `security()` over the 180 attacked runs (injection succeeded); **lower with DFC = the defense
works**. Utility = clean-task completion. Results are written to this folder.

**Speed:** the agent/DFC path is heavy synchronous Python, and the GIL serializes it — so a plain thread
pool plateaus around ~16 min no matter how many threads. Instead we fan out across **processes** (multiple
GILs → the per-task glue parallelizes across cores), each running a **thread pool** (LLM-call overlap).
That takes the DFC run to roughly **~6-9 min**. Retries + `RERUN=False` resume absorb Bedrock throttling
and let you stop/restart freely.""")

md("## Setup")
code("""import os, json, time, queue, importlib, sys, multiprocessing as mp
from pathlib import Path

CWD = Path.cwd()
REPO = next((p for p in [CWD, *CWD.parents] if (p / "runs").is_dir()), None)
assert REPO is not None, f"Could not locate runs/. Tried from {CWD}"
RUNS = REPO / "runs"
OUTDIR = REPO / "notebooks" / "final_runs" / "Qwen_3"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR = OUTDIR / "logs"; LOGDIR.mkdir(exist_ok=True)

# The agent/DFC machinery is heavy Python glue that the GIL serializes, so we run it across PROCESSES (see
# Config). All of that lives in _mp_worker.py next to this notebook -- a real module so Windows 'spawn' can
# re-import it in the child processes (a function defined in the notebook itself would NOT be importable).
sys.path.insert(0, str(OUTDIR))
import _mp_worker; importlib.reload(_mp_worker)   # reload so edits to the worker take effect without a kernel restart

print("repo:", REPO, "| outdir:", OUTDIR, "| cores:", os.cpu_count())""")

md("## Config")
code("""SUITE  = "shopping"
VER    = "v1.2"
ATTACK = "important_instructions"

AGENT      = "qwen3-235b"                       # LiteLLM Bedrock Converse profile (us-west-2)
DFC_MODEL  = "us.anthropic.claude-opus-4-8"     # DFC fact-extraction + policy generation (us-east-1)
CLASSIFIER = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # DFC sink classifier (admission control)

# Full suite.
USER_TASKS = [f"user_task_{i}" for i in range(20)]
INJ_TASKS  = [f"injection_task_{i}" for i in range(9)]

# ---- Parallelism: PROCESSES x THREADS (the speed knob) ----
# The wall isn't the LLM I/O (that overlaps fine) -- it's the per-task PYTHON glue (DFC validation, policy
# parsing, logging), which the GIL serializes. So pure threads plateau ~16 min regardless of count. Multiple
# PROCESSES give multiple GILs (the glue parallelizes across cores); THREADS-per-process give I/O overlap.
# ~12 procs x 8 threads (= ~96 concurrent LLM calls) takes the DFC run from ~16 min to roughly ~6-9 min.
N_PROC  = min(12, os.cpu_count() or 12)   # one GIL per process, ~= physical cores
THREADS = 8                                # concurrent LLM calls per process
RERUN   = True                             # always fresh; False = resume (worker skips finished tasks in runs/)
OVERALL_BUDGET = 7200                       # s; stop waiting after this and report whatever finished

# Passed to every worker process (must be plain picklable data -- no live objects).
CONFIG = {
    "agent": AGENT, "dfc_model": DFC_MODEL, "classifier_model": CLASSIFIER,
    "suite": SUITE, "ver": VER, "attack": ATTACK, "runs": str(RUNS), "rerun": RERUN,
    "bedrock_max_retries": 10, "litellm_retries": 10,
    "agent_io_log": str(LOGDIR / "agent_io.jsonl"),   # workers write <path>.<pid>, merged after each run
    "run_log":      str(LOGDIR / "run.log"),
    "errors_log":   str(LOGDIR / "errors.log"),
}

def run_condition(cond, n_proc=N_PROC, threads=THREADS):
    jobs  = [(cond, "clean",  ut, None) for ut in USER_TASKS]
    jobs += [(cond, "attack", ut, inj)  for ut in USER_TASKS for inj in INJ_TASKS]
    out = OUTDIR / f"{cond}_results.jsonl"; out.write_text("", encoding="utf-8")   # fresh file for this run

    mgr = mp.Manager(); in_q = mgr.Queue(); out_q = mgr.Queue()
    for j in jobs:
        in_q.put(j)
    for _ in range(n_proc * threads):
        in_q.put(None)                              # one sentinel per thread -> clean shutdown when drained
    procs = [mp.Process(target=_mp_worker.worker, args=(in_q, out_q, CONFIG, threads)) for _ in range(n_proc)]

    results, t0, deadline = [], time.time(), time.time() + OVERALL_BUDGET
    for p in procs:
        p.start()
    while len(results) < len(jobs):
        try:
            r = out_q.get(timeout=120)              # a worker emitted a finished result
        except queue.Empty:
            if all(not p.is_alive() for p in procs):
                print(f"  !! all workers exited early with {len(results)}/{len(jobs)} done"); break
            if time.time() > deadline:
                print(f"  !! OVERALL_BUDGET {OVERALL_BUDGET}s hit; {len(results)}/{len(jobs)} done"); break
            continue                                # workers still alive, keep waiting
        results.append(r)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\\n")
        n = len(results)
        if n % 10 == 0 or r.get("error") or n == len(jobs):
            errs = sum(1 for x in results if x.get("error"))
            tag = f"{r['ut']}{('/' + r['inj']) if r['inj'] else ''}"
            print(f"  [{n:3}/{len(jobs)} {time.time()-t0:6.0f}s] errors={errs}  last={tag} sec={r['security']}")
    for p in procs:
        p.terminate()                               # stop any idle/blocked workers, then reap
    for p in procs:
        p.join(timeout=10)
    _mp_worker.merge_logs(LOGDIR)                    # fold per-process log shards (<base>.<pid>) back together
    dt = time.time() - t0
    print(f"  {cond}: {len(results)}/{len(jobs)} jobs in {dt:.0f}s ({dt/60:.1f} min) -> {out}")
    return results

print(f"agent={AGENT}  dfc={DFC_MODEL}  |  {N_PROC} procs x {THREADS} threads = {N_PROC*THREADS} concurrent LLM calls")
print(f"per condition: {len(USER_TASKS)} clean + {len(USER_TASKS)*len(INJ_TASKS)} attacked = {len(USER_TASKS)*(1+len(INJ_TASKS))} jobs")""")

md("## 1. Baseline (no defense)")
code("""print('=== BASELINE (no defense) ===')
BASELINE = run_condition('baseline')""")

md("## 2. DFC")
code("""print('=== DFC (admission control + feedback) ===')
DFC = run_condition('dfc')""")

md("## Scoreboard")
code("""def summarize(results):
    att = [r for r in results if r["kind"] == "attack"]
    sec = [r["security"] for r in att if r["security"] is not None]
    util = [r["utility"] for r in results if r["kind"] == "clean" and r["utility"] is not None]
    errs = sum(1 for r in results if r["error"])
    return {
        "ASR": (sum(sec)/len(sec)) if sec else None, "n_att": len(sec),
        "utility": (sum(util)/len(util)) if util else None, "n_clean": len(util),
        "errors": errs,
    }

b, d = summarize(BASELINE), summarize(DFC)
print("=" * 70)
print(f"SHOPPING (full) | agent=qwen3-235b | dfc-policy=opus-4-8")
print("=" * 70)
for label, m in [("baseline", b), ("dfc", d)]:
    a = f"{m['ASR']*100:5.1f}%" if m['ASR'] is not None else "  n/a"
    u = f"{m['utility']*100:5.1f}%" if m['utility'] is not None else "  n/a"
    print(f"  {label:9}  ASR={a} (n={m['n_att']})   utility(clean)={u} (n={m['n_clean']})   errors={m['errors']}")
print("=" * 70)
if b['ASR'] is not None and d['ASR'] is not None:
    print(f"ASR: baseline {b['ASR']*100:.1f}% -> dfc {d['ASR']*100:.1f}%   (delta {(d['ASR']-b['ASR'])*100:+.1f} pts, negative = DFC blocked more)")
summary = {"agent": AGENT, "dfc_model": DFC_MODEL, "suite": SUITE, "baseline": b, "dfc": d}
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("wrote", OUTDIR / "summary.json")""")

md("""## Where injections succeeded / were blocked (detail)""")
code("""import collections
def hits(results):
    return sorted((r["ut"], r["inj"]) for r in results if r["kind"]=="attack" and r["security"] is True)
bh, dh = set(hits(BASELINE)), set(hits(DFC))
print(f"baseline: {len(bh)} successful injections")
print(f"dfc:      {len(dh)} successful injections")
blocked_by_dfc = sorted(bh - dh)
newly_open    = sorted(dh - bh)
print(f"\\nDFC fixed (succeeded in baseline, blocked by DFC): {len(blocked_by_dfc)}")
for ut, inj in blocked_by_dfc: print(f"   {ut} x {inj}")
if newly_open:
    print(f"\\n!! succeeded only WITH dfc (regression): {len(newly_open)}")
    for ut, inj in newly_open: print(f"   {ut} x {inj}")""")

md("""## Notes

- **Parallelism (`N_PROC` x `THREADS`):** the run fans out across **processes** (multiple GILs → the per-task
  Python glue parallelizes across cores) each running a **thread pool** (I/O overlap). Pure threads plateau
  ~16 min no matter the count because the GIL serializes the glue; ~12 procs x 8 threads cuts the DFC run to
  roughly ~6-9 min. If you have more/fewer cores, adjust `N_PROC`; raise `THREADS` only if Bedrock isn't
  throttling. The worker (`_mp_worker.py`) is reloaded each Setup run, so edits to it apply without a kernel
  restart — but edits to the **DFC framework** (e.g. `prompts.py`) need a **kernel restart** to re-import.
- **Resume:** `RERUN=False` makes the worker skip tasks already in `runs/`. Stop/restart anytime — finished
  tasks are reused, only missing ones run. `RERUN=True` re-runs everything fresh.
- **Robustness:** results stream back over a queue; if a worker process dies, the runner notices (liveness
  check + 120s get-timeout) and reports partial progress instead of hanging. `OVERALL_BUDGET` caps total wait.
- **Outputs:** per-condition results in `baseline_results.jsonl` / `dfc_results.jsonl`, the headline in
  `summary.json`, all in this folder. Generated policies are in `runs/<pipeline>/.../*_dfc/metadata.json`
  and `runs/policy_ledger.jsonl` (browse with `notebooks/_policy_catalog.py`).
- **Everything is logged** under `logs/` in this folder (workers write per-process `<file>.<pid>` shards,
  merged back together at the end of each run):
  - `agent_io.jsonl` — every Qwen/litellm call: raw messages sent, the model's response + tool calls,
    token usage, latency, and any failure.
  - `run.log` — all WARNING+ from litellm / anthropic / botocore (throttles, retries, backoffs).
  - `errors.log` — full Python tracebacks for any job that errored (the result row keeps a short repr).
  - The DFC side (facts, generated/admitted policies, classifier verdicts, feedback rounds, validation
    blocks) is in each task's `runs/.../*_dfc/{metadata.json,events.jsonl}`.
- Built by `notebooks/_build_qwen_full.py` — edit the script and rebuild rather than hand-editing the
  `.ipynb`. The worker logic is in `notebooks/final_runs/Qwen_3/_mp_worker.py`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).parent / "final_runs" / "Qwen_3" / "qwen3_shopping_full.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
