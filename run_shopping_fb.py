"""Drive the full gpt-oss shopping benchmark (baseline + dfc) with the updated DFC feedback code, and report
ASR / utility(clean) / utility(under attack) for both conditions. Mirrors the notebook's run_condition + scoreboard.
"""
import os, sys, json, time, queue
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, "notebooks/final_runs/Qwen_3")
import _mp_worker  # noqa: E402

AGENT, SUITE, VER = "gpt-oss-120b", "shopping", "v1.2"
RUN_DIR = Path("runs_shopping_fb"); LOGDIR = RUN_DIR / "logs"
RUN_DIR.mkdir(exist_ok=True); LOGDIR.mkdir(exist_ok=True)
N_PROC, THREADS = 6, 6

CONFIG = {
    "agent": AGENT, "dfc_model": "us.anthropic.claude-opus-4-8",
    "classifier_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "suite": SUITE, "ver": VER, "attack": "important_instructions", "runs": "runs",
    "rerun": True, "bedrock_max_retries": 10, "litellm_retries": 10,
    "semantic_judge": True, "judge_mode": "injection_detect", "injection_quarantine": False,
    "run_log": str(LOGDIR / "run.log"), "errors_log": str(LOGDIR / "errors.log"),
}


def run_condition(cond, user_tasks, inj_tasks):
    jobs = [(cond, "clean", ut, None) for ut in user_tasks]
    jobs += [(cond, "attack", ut, inj) for ut in user_tasks for inj in inj_tasks]
    out = RUN_DIR / f"{cond}_results.jsonl"; out.write_text("", encoding="utf-8")
    mgr = mp.Manager(); in_q = mgr.Queue(); out_q = mgr.Queue()
    for j in jobs:
        in_q.put(j)
    for _ in range(N_PROC * THREADS):
        in_q.put(None)
    procs = [mp.Process(target=_mp_worker.worker, args=(in_q, out_q, CONFIG, THREADS)) for _ in range(N_PROC)]
    results, t0 = [], time.time()
    for p in procs:
        p.start()
    while len(results) < len(jobs):
        try:
            r = out_q.get(timeout=240)
        except queue.Empty:
            if all(not p.is_alive() for p in procs):
                print(f"  !! workers exited early {len(results)}/{len(jobs)}", flush=True); break
            continue
        results.append(r)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        if len(results) % 20 == 0:
            errs = sum(1 for x in results if x.get("error"))
            print(f"  {cond}: {len(results)}/{len(jobs)} {time.time()-t0:.0f}s errs={errs}", flush=True)
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=10)
    _mp_worker.merge_logs(LOGDIR)
    print(f"{cond} DONE: {len(results)}/{len(jobs)} in {time.time()-t0:.0f}s", flush=True)
    return results


def summarize(results):
    att = [r for r in results if r["kind"] == "attack"]
    sec = [r["security"] for r in att if r["security"] is not None]
    util = [r["utility"] for r in results if r["kind"] == "clean" and r["utility"] is not None]
    util_att = [r["utility"] for r in att if r["utility"] is not None]
    errs = sum(1 for r in results if r.get("error"))
    f = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"ASR": f(sec), "n_att": len(sec), "utility": f(util), "n_clean": len(util),
            "utility_attack": f(util_att), "n_att_util": len(util_att), "errors": errs}


if __name__ == "__main__":
    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite(VER, SUITE)
    USER_TASKS, INJ_TASKS = list(suite.user_tasks), list(suite.injection_tasks)
    print(f"shopping | {AGENT} | {len(USER_TASKS)}x{len(INJ_TASKS)} attacked + {len(USER_TASKS)} clean per cond", flush=True)
    print("=== BASELINE ===", flush=True); B = run_condition("baseline", USER_TASKS, INJ_TASKS)
    print("=== DFC (updated feedback) ===", flush=True); D = run_condition("dfc", USER_TASKS, INJ_TASKS)
    b, d = summarize(B), summarize(D)
    pct = lambda x: f"{x*100:5.1f}%" if x is not None else "  n/a"
    print("=" * 84, flush=True)
    for label, m in [("baseline", b), ("dfc", d)]:
        print(f"  {label:9} ASR={pct(m['ASR'])} (n={m['n_att']})  utility(clean)={pct(m['utility'])} (n={m['n_clean']})  "
              f"utility(UNDER ATTACK)={pct(m['utility_attack'])} (n={m['n_att_util']})  errors={m['errors']}", flush=True)
    print("=" * 84, flush=True)
    print(f"ASR: {pct(b['ASR'])} -> {pct(d['ASR'])}  (delta {(d['ASR']-b['ASR'])*100:+.1f} pts)", flush=True)
    print(f"baseline_utility_under_attack = {pct(b['utility_attack'])}", flush=True)
    print(f"dfc_utility_under_attack      = {pct(d['utility_attack'])}  "
          f"(delta {(d['utility_attack']-b['utility_attack'])*100:+.1f} pts)", flush=True)
    RUN_DIR.joinpath("scoreboard.json").write_text(json.dumps({"baseline": b, "dfc": d}, indent=2), encoding="utf-8")
