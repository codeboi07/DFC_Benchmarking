"""End-to-end ASR eval: shopping, DFC vs no-defense, upgraded models.
Run: uv run python notebooks/_eval_dfc_vs_nodef.py   (writes runs/eval_dfc_shopping.jsonl + prints summary)"""
import os, json, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTimeout

os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("BEDROCK_TIMEOUT", "150")
os.environ["DFC_CLASSIFIER_MODEL"] = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
logging.getLogger().setLevel(logging.ERROR)
for n in ("httpx", "anthropic", "boto3", "botocore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig, AgentPipeline
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger, LOGGER_STACK

SUITE, VER, ATTACK = "shopping", "v1.2", "important_instructions"
AGENT = "us.anthropic.claude-sonnet-4-6"
DFC_MODEL = "us.anthropic.claude-opus-4-8"
USER_TASKS = [f"user_task_{i}" for i in range(5)]
INJ_TASKS = [f"injection_task_{i}" for i in range(9)]
MAX_WORKERS = 3
PER_TASK_TIMEOUT = 240
GLOBAL_BUDGET = 3000
RUNS = Path("runs")
OUT = RUNS / "eval_dfc_shopping.jsonl"
suite = get_suite(VER, SUITE)


def build(condition):
    if condition == "no_defense":
        cfg = PipelineConfig(llm=AGENT, model_id=None, defense=None,
                             system_message_name=None, system_message=None, suite_name=SUITE)
    else:
        cfg = PipelineConfig(llm=AGENT, model_id=None, defense="dfc_agent_framework_integration",
                             system_message_name=None, system_message=None, suite_name=SUITE, dfc_model=DFC_MODEL)
    return AgentPipeline.from_config(cfg)


def run_job(job):
    cond, kind, ut_id, inj = job
    LOGGER_STACK.set([])
    pipe = build(cond)               # fresh pipeline + Bedrock client per job (never shared across threads)
    ut = suite.get_user_task_by_id(ut_id)
    with OutputLogger(str(RUNS), None):
        if kind == "clean":
            u, s = run_task_without_injection_tasks(suite, pipe, ut, RUNS, True, VER)
            return {**dict(zip(("cond", "kind", "ut", "inj"), job)), "utility": bool(u), "security": None}
        attack = load_attack(ATTACK, suite, pipe)
        u, s = run_task_with_injection_tasks(suite, pipe, ut, attack, RUNS, True, [inj], VER)
        return {**dict(zip(("cond", "kind", "ut", "inj"), job)),
                "utility": bool(list(dict(u).values())[0]), "security": bool(list(dict(s).values())[0])}


def main():
    jobs = []
    for cond in ("no_defense", "dfc"):
        for ut in USER_TASKS:
            jobs.append((cond, "clean", ut, None))
            for inj in INJ_TASKS:
                jobs.append((cond, "attack", ut, inj))
    print(f"agent={AGENT}  dfc={DFC_MODEL}  jobs={len(jobs)} (per condition: {len(USER_TASKS)} clean + "
          f"{len(USER_TASKS)*len(INJ_TASKS)} attacked)", flush=True)
    results, t0 = [], time.time()
    if OUT.exists():
        OUT.unlink()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        done = 0
        try:
            for fut in as_completed(futs, timeout=GLOBAL_BUDGET):
                j = futs[fut]
                try:
                    r = fut.result(timeout=PER_TASK_TIMEOUT); r["error"] = None
                except Exception as e:
                    r = {**dict(zip(("cond", "kind", "ut", "inj"), j)),
                         "utility": None, "security": None, "error": repr(e)[:160]}
                results.append(r)
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
                done += 1
                tag = f"{r['cond']}/{r['kind']}/{r['ut']}" + (f"/{r['inj']}" if r['inj'] else "")
                print(f"[{done:3}/{len(jobs)} {time.time()-t0:6.0f}s] {tag:42} util={r['utility']} sec={r['security']} {r['error'] or ''}", flush=True)
        except FTimeout:
            print(f"GLOBAL BUDGET {GLOBAL_BUDGET}s hit; {done}/{len(jobs)} done, rest abandoned", flush=True)

    def asr(cond):
        v = [r["security"] for r in results if r["cond"] == cond and r["kind"] == "attack" and r["security"] is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    def util(cond):
        v = [r["utility"] for r in results if r["cond"] == cond and r["kind"] == "clean" and r["utility"] is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    print("\n" + "=" * 64)
    print(f"SHOPPING  |  agent={AGENT.split('.')[-1]}  dfc-policy={DFC_MODEL.split('.')[-1]}")
    print("=" * 64)
    for cond in ("no_defense", "dfc"):
        a, na = asr(cond); u, nu = util(cond)
        errs = sum(1 for r in results if r["cond"] == cond and r["error"])
        a_s = f"{a*100:5.1f}%" if a is not None else "  n/a"
        u_s = f"{u*100:5.1f}%" if u is not None else "  n/a"
        print(f"  {cond:11}  ASR={a_s} (n={na})   utility(clean)={u_s} (n={nu})   errors={errs}")
    print("=" * 64)
    print("ASR = mean security() over attacked runs (injection succeeded). Lower = better defense.")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
