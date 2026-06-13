#!/usr/bin/env python3
"""Generate DFC sink policies for tau3/tau2-bench and run baseline/DFC evals.

This script targets the current tau-three implementation in
https://github.com/sierra-research/tau2-bench. It is intentionally conservative:
it generates an auditable sink map from domain tools, installs a lightweight DFC
agent adapter into the tau2 checkout, runs baseline tau2 evaluations, optionally
runs DFC evaluations, and compares utility side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/sierra-research/tau2-bench.git"
DEFAULT_DOMAINS = ("retail", "airline", "telecom", "banking_knowledge")
DEFAULT_MODEL = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_USER_MODEL = DEFAULT_MODEL
DEFAULT_RETRIEVAL_CONFIG = "bm25"
DFC_UV_PACKAGES = ("duckdb", "data-flow-control", "boto3")


@dataclass(frozen=True)
class RunNames:
    baseline: str
    dfc: str


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def capture(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, env=env, text=True)


def ensure_repo(repo_dir: Path, repo_url: str, ref: str | None) -> None:
    if repo_dir.exists():
        print(f"Using existing tau2 checkout: {repo_dir}")
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(repo_dir)])
    if ref:
        run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
        run(["git", "checkout", ref], cwd=repo_dir)


def ensure_uv() -> None:
    if shutil.which("uv") is None:
        raise SystemExit("uv is required. Install from https://docs.astral.sh/uv/ then rerun.")


def uv_sync(repo_dir: Path, include_knowledge: bool) -> None:
    cmd = ["uv", "sync"]
    if include_knowledge:
        cmd += ["--extra", "knowledge"]
    run(cmd, cwd=repo_dir)
    run(["uv", "pip", "install", "duckdb", "data-flow-control", "boto3"], cwd=repo_dir)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def generate_policy_map(repo_dir: Path, domains: list[str], output_path: Path) -> dict[str, Any]:
    helper = repo_dir / "_dfc_generate_tau3_policy_map.py"
    write_text(
        helper,
        r'''
        from __future__ import annotations

        import json
        import re
        import sys

        from tau2.runner import build_environment

        EFFECTFUL_VERBS = (
            "add", "apply", "book", "buy", "cancel", "change", "close", "confirm",
            "create", "delete", "disable", "enable", "exchange", "modify", "open",
            "order", "pay", "purchase", "refund", "remove", "reserve", "reset",
            "schedule", "send", "set", "submit", "suspend", "transfer", "update",
            "upgrade", "downgrade", "replace", "escalate",
        )
        READ_ONLY_PREFIXES = (
            "get", "list", "search", "find", "lookup", "check", "calculate",
            "retrieve", "view", "read", "query", "browse", "load", "fetch",
        )
        SENSITIVE_ARG_RE = re.compile(
            r"(email|recipient|to|cc|bcc|phone|address|account|iban|routing|card|"
            r"amount|price|payment|password|ssn|pin|token|credential|order|reservation|"
            r"booking|ticket|passenger|customer|user|plan|service|device|line|subscription|"
            r"flight|item|product|quantity|date|time|name|id)$",
            re.I,
        )
        BENIGN_ARG_RE = re.compile(r"^(query|q|search|keyword|keywords|limit|page|sort|filter)$", re.I)

        def schema_props(tool):
            schema = getattr(tool, "openai_schema", {}) or {}
            params = schema.get("function", {}).get("parameters", {}) or {}
            return params.get("properties", {}) or {}, set(params.get("required", []) or [])

        def is_effectful(name: str, description: str) -> bool:
            lname = name.lower()
            if lname.startswith(READ_ONLY_PREFIXES):
                return False
            if lname.startswith(EFFECTFUL_VERBS):
                return True
            text = f"{name} {description}".lower()
            return any(f" {verb} " in f" {text} " for verb in EFFECTFUL_VERBS)

        def scalarish(prop: dict) -> bool:
            typ = prop.get("type")
            if typ in {"string", "integer", "number", "boolean"}:
                return True
            if typ == "array":
                item_type = (prop.get("items") or {}).get("type")
                return item_type in {"string", "integer", "number", "boolean"}
            return "anyOf" in prop or "oneOf" in prop

        domains = sys.argv[1].split(",")
        out = {
            "version": 1,
            "generator": "scripts/tau3_dfc_benchmark.py",
            "notes": [
                "Generated by effectful-tool and sensitive-argument heuristics.",
                "Review before final runs; tighten or loosen per domain as needed.",
            ],
            "domains": {},
        }
        for domain in domains:
            env_kwargs = {}
            if domain == "banking_knowledge":
                env_kwargs["retrieval_variant"] = "bm25"
            env = build_environment(domain, env_kwargs=env_kwargs)
            specs = []
            for tool in env.get_tools():
                name = tool.name
                desc = getattr(tool, "description", "") or ""
                props, required = schema_props(tool)
                if not is_effectful(name, desc):
                    continue
                selected = [
                    key for key, prop in props.items()
                    if scalarish(prop) and SENSITIVE_ARG_RE.search(key) and not BENIGN_ARG_RE.search(key)
                ]
                if not selected:
                    selected = [
                        key for key, prop in props.items()
                        if key in required and scalarish(prop) and not BENIGN_ARG_RE.search(key)
                    ]
                if selected:
                    specs.append({
                        "tool": name,
                        "columns": sorted(dict.fromkeys(selected)),
                        "description": desc,
                    })
            out["domains"][domain] = sorted(specs, key=lambda x: x["tool"])
        print(json.dumps(out, indent=2, sort_keys=True))
        ''',
    )
    cmd = ["uv", "run"]
    for package in DFC_UV_PACKAGES:
        cmd += ["--with", package]
    cmd += ["python", str(helper), ",".join(domains)]
    text = capture(cmd, cwd=repo_dir)
    data = json.loads(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote generated policy map: {output_path}")
    return data


def install_adapter(repo_dir: Path, policy_map_path: Path) -> None:
    adapter_dir = repo_dir / "src" / "tau2" / "agent" / "dfc_adapter"
    write_text(adapter_dir / "__init__.py", "from .llm_agent import create_dfc_llm_agent\n")
    rel_policy_map = os.path.relpath(policy_map_path, adapter_dir).replace("\\", "/")

    write_text(
        adapter_dir / "policies.py",
        f'''
        from __future__ import annotations

        import json
        from dataclasses import dataclass
        from pathlib import Path

        from data_flow_control import Policy

        POLICY_MAP_PATH = Path(__file__).resolve().parent / "{rel_policy_map}"


        @dataclass(frozen=True)
        class SinkSpec:
            tool: str
            columns: tuple[str, ...]


        def _load() -> dict:
            return json.loads(POLICY_MAP_PATH.read_text(encoding="utf-8"))


        def sink_specs(domain: str | None) -> tuple[SinkSpec, ...]:
            raw = (_load().get("domains") or {{}}).get(domain or "", [])
            return tuple(SinkSpec(s["tool"], tuple(s["columns"])) for s in raw)


        def sink_tools(domain: str | None) -> set[str]:
            return {{s.tool for s in sink_specs(domain)}}


        def sink_columns(domain: str | None, tool: str) -> tuple[str, ...]:
            cols: list[str] = []
            for spec in sink_specs(domain):
                if spec.tool == tool:
                    cols.extend(spec.columns)
            return tuple(dict.fromkeys(cols))


        def grounding_policy() -> Policy:
            return Policy.from_pgn("""
            SOURCE REQUIRED trusted AS T SINK sink_input AS S
            CONSTRAINT S.value = T.value
            ON FAIL REMOVE
            DESCRIPTION Sensitive tool arguments must be grounded in user-originated trusted values.
            """)
        ''',
    )

    write_text(
        adapter_dir / "materialize.py",
        r'''
        from __future__ import annotations

        import json
        import re
        from typing import Any

        EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        IBAN = re.compile(r"\b[A-Z]{2}\d{6,}\b")
        IDENT = re.compile(r"\b(?:order|reservation|booking|ticket|passenger|customer|account|user|line|device|plan|flight|item|product)[-_ ]?[A-Za-z0-9]{2,}\b", re.I)
        QUOTED = re.compile(r'"([^"]{2,})"|\'([^\']{2,})\'')


        def as_dict(obj: Any) -> dict[str, Any]:
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


        def _message_text(messages: list[Any]) -> str:
            parts: list[str] = []
            for message in messages or []:
                d = as_dict(message)
                if d.get("role") == "user":
                    parts.append(str(d.get("content", "")))
            return "\n".join(parts)


        def build_trusted(messages: list[Any]) -> list[tuple[str]]:
            text = _message_text(messages)
            values: set[str] = set()
            values.update(EMAIL.findall(text))
            values.update(IBAN.findall(text))
            values.update(m.group(0).replace(" ", "_") for m in IDENT.finditer(text))
            for quoted in QUOTED.findall(text):
                token = quoted[0] or quoted[1]
                if token:
                    values.add(token)
            # Include short structured tokens from the user text. Avoid loose prose.
            for token in re.findall(r"\b[A-Z0-9][A-Z0-9_-]{2,}\b", text):
                values.add(token)
            return [(v,) for v in sorted(values)]


        def parse_args(args: Any) -> dict[str, Any]:
            if isinstance(args, dict):
                return args
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
            return as_dict(args)


        def candidate_values(args: dict[str, Any], columns: tuple[str, ...]) -> list[str]:
            out: list[str] = []
            for col in columns:
                value = args.get(col)
                if isinstance(value, list):
                    out.extend(str(v) for v in value if v is not None)
                elif value is not None:
                    out.append(str(value))
            return out
        ''',
    )

    write_text(
        adapter_dir / "session.py",
        r'''
        from __future__ import annotations

        import duckdb
        from data_flow_control import dfc

        from .materialize import build_trusted, candidate_values
        from .policies import grounding_policy, sink_columns, sink_tools


        class DFCSession:
            def __init__(self, domain: str | None, messages):
                self.domain = domain
                self.guarded_tools = sink_tools(domain)
                self.raw = duckdb.connect()
                self.raw.execute("CREATE OR REPLACE TABLE trusted(value TEXT)")
                trusted = build_trusted(list(messages or []))
                if trusted:
                    self.raw.executemany("INSERT INTO trusted VALUES (?)", trusted)
                self.raw.execute("CREATE OR REPLACE TABLE sink_input(value TEXT)")
                self.conn = dfc(self.raw)
                self.conn.register_policy(grounding_policy())

            def allows_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str | None]:
                if tool_name not in self.guarded_tools:
                    return True, None
                values = candidate_values(args, sink_columns(self.domain, tool_name))
                if not values:
                    return True, None
                self.raw.execute("DELETE FROM sink_input")
                self.raw.executemany("INSERT INTO sink_input VALUES (?)", [(v,) for v in values])
                allowed = {row[0] for row in self.conn.sql("SELECT value FROM sink_input").fetchall()}
                blocked = sorted(set(values) - allowed)
                if blocked:
                    return False, f"blocked by DFC: {tool_name} used untrusted value(s) {blocked}"
                return True, None
        ''',
    )

    write_text(
        adapter_dir / "llm_agent.py",
        r'''
        from __future__ import annotations

        import os

        from tau2.agent import LLMAgent
        from tau2.data_model.message import AssistantMessage

        from .materialize import as_dict, parse_args
        from .session import DFCSession


        class DFCLLMAgent(LLMAgent):
            def __init__(self, *args, domain_name: str | None = None, **kwargs):
                super().__init__(*args, **kwargs)
                self.domain_name = domain_name

            def generate_next_message(self, message, state):
                # Use the parent internal method so we can append the filtered response.
                response = self._generate_next_message(message, state)
                tool_calls = response.tool_calls or []
                if not tool_calls:
                    state.messages.append(response)
                    return response, state

                session = DFCSession(self.domain_name, state.messages)
                allowed_calls = []
                blocked_messages = []
                for call in tool_calls:
                    call_dict = as_dict(call)
                    name = call_dict.get("name")
                    args = parse_args(call_dict.get("arguments") or {})
                    ok, reason = session.allows_tool_call(str(name), args)
                    if ok:
                        allowed_calls.append(call)
                    else:
                        blocked_messages.append(reason or f"blocked by DFC: {name}")

                if not blocked_messages:
                    state.messages.append(response)
                    return response, state

                if allowed_calls:
                    response = response.model_copy(update={"tool_calls": allowed_calls})
                else:
                    response = AssistantMessage(role="assistant", content="; ".join(blocked_messages))
                state.messages.append(response)
                return response, state


        def create_dfc_llm_agent(tools, domain_policy, **kwargs):
            task = kwargs.get("task")
            domain_name = (
                getattr(task, "domain", None)
                or kwargs.get("domain_name")
                or os.environ.get("TAU2_DFC_DOMAIN")
            )
            return DFCLLMAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args") or {},
                domain_name=domain_name,
            )
        ''',
    )

    registry = repo_dir / "src" / "tau2" / "registry.py"
    text = registry.read_text(encoding="utf-8")
    registration = (
        "\n# DFC adapter installed by scripts/tau3_dfc_benchmark.py\n"
        "from tau2.agent.dfc_adapter import create_dfc_llm_agent\n"
        "registry.register_agent_factory(create_dfc_llm_agent, \"dfc_llm_agent\")\n"
    )
    if "dfc_llm_agent" not in text:
        registry.write_text(text + registration, encoding="utf-8")
    print(f"Installed DFC adapter in {adapter_dir}")


def run_tau(
    repo_dir: Path,
    domain: str,
    *,
    agent: str,
    agent_model: str,
    user_model: str,
    num_trials: int,
    num_tasks: int | None,
    task_ids: list[str] | None,
    max_concurrency: int,
    seed: int,
    save_to: str,
    retrieval_config: str,
    extra_args: list[str],
) -> None:
    cmd = ["uv", "run"]
    for package in DFC_UV_PACKAGES:
        cmd += ["--with", package]
    cmd += [
        "tau2", "run",
        "--domain", domain,
        "--agent", agent,
        "--agent-llm", agent_model,
        "--user-llm", user_model,
        "--num-trials", str(num_trials),
        "--max-concurrency", str(max_concurrency),
        "--seed", str(seed),
        "--save-to", save_to,
        "--auto-resume",
    ]
    if num_tasks is not None:
        cmd += ["--num-tasks", str(num_tasks)]
    if task_ids:
        cmd += ["--task-ids", *task_ids]
    if domain == "banking_knowledge":
        cmd += ["--retrieval-config", retrieval_config]
    cmd += extra_args
    env = os.environ.copy()
    env["TAU2_DFC_DOMAIN"] = domain
    run(cmd, cwd=repo_dir, env=env)


def load_results(repo_dir: Path, save_to: str) -> list[dict[str, Any]]:
    path = repo_dir / "data" / "simulations" / save_to / "results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    sims = data.get("simulations") or []
    # Directory format fallback.
    if not sims:
        sim_dir = path.parent / "simulations"
        if sim_dir.exists():
            for item in sim_dir.glob("*.json"):
                sims.append(json.loads(item.read_text(encoding="utf-8")))
    return sims


def reward_of(sim: dict[str, Any]) -> float | None:
    info = sim.get("reward_info") or {}
    reward = info.get("reward", sim.get("reward"))
    return None if reward is None else float(reward)


def task_id_of(sim: dict[str, Any]) -> str:
    task = sim.get("task") or {}
    return str(task.get("id") or sim.get("task_id") or sim.get("id") or "")


def summarize(sims: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [r for r in (reward_of(sim) for sim in sims) if r is not None]
    successes = [1 if r >= 1 - 1e-6 else 0 for r in rewards]
    return {
        "n": len(rewards),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "pass_rate": sum(successes) / len(successes) if successes else 0.0,
    }


def analyze(repo_dir: Path, domains: list[str], names: RunNames, out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for domain in domains:
        base_name = f"{names.baseline}_{domain}"
        dfc_name = f"{names.dfc}_{domain}"
        base = load_results(repo_dir, base_name)
        dfc = load_results(repo_dir, dfc_name)
        bs = summarize(base)
        ds = summarize(dfc)
        rows.append({
            "domain": domain,
            "baseline_n": bs["n"],
            "dfc_n": ds["n"],
            "baseline_avg_reward": bs["avg_reward"],
            "dfc_avg_reward": ds["avg_reward"],
            "delta_avg_reward": ds["avg_reward"] - bs["avg_reward"],
            "baseline_pass_rate": bs["pass_rate"],
            "dfc_pass_rate": ds["pass_rate"],
            "delta_pass_rate": ds["pass_rate"] - bs["pass_rate"],
        })
        base_by_task: dict[str, list[float]] = {}
        dfc_by_task: dict[str, list[float]] = {}
        for sim in base:
            tid = task_id_of(sim)
            r = reward_of(sim)
            if tid and r is not None:
                base_by_task.setdefault(tid, []).append(r)
        for sim in dfc:
            tid = task_id_of(sim)
            r = reward_of(sim)
            if tid and r is not None:
                dfc_by_task.setdefault(tid, []).append(r)
        for tid in sorted(set(base_by_task) | set(dfc_by_task)):
            bvals = base_by_task.get(tid, [])
            dvals = dfc_by_task.get(tid, [])
            bavg = sum(bvals) / len(bvals) if bvals else 0.0
            davg = sum(dvals) / len(dvals) if dvals else 0.0
            task_rows.append({
                "domain": domain,
                "task_id": tid,
                "baseline_n": len(bvals),
                "dfc_n": len(dvals),
                "baseline_avg_reward": bavg,
                "dfc_avg_reward": davg,
                "delta_avg_reward": davg - bavg,
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [("summary.csv", rows), ("by_task.csv", task_rows)]:
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else ["empty"])
            writer.writeheader()
            writer.writerows(data)
    print("\nSide-by-side utility summary")
    for row in rows:
        print(
            f"{row['domain']:18s} "
            f"base={row['baseline_pass_rate']:.3f} "
            f"dfc={row['dfc_pass_rate']:.3f} "
            f"delta={row['delta_pass_rate']:+.3f} "
            f"n={row['baseline_n']}/{row['dfc_n']}"
        )
    print(f"Wrote analysis: {out_dir / 'summary.csv'} and {out_dir / 'by_task.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/tau2-bench"))
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default=None, help="Optional git branch/tag/SHA to checkout.")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--agent-model", default=DEFAULT_MODEL)
    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Trials per task. Defaults to 1 for the requested full-benchmark run.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Optional smoke-test task limit. Omit to run every task.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Explicitly run every task. This is already the default.",
    )
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--retrieval-config", default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--baseline-prefix", default="baseline_bedrock_sonnet")
    parser.add_argument("--dfc-prefix", default="dfc_bedrock_sonnet")
    parser.add_argument("--policy-map", type=Path, default=Path("generated/tau3/generated_tau3_dfc_policies.json"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("generated/tau3/analysis"))
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-dfc", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run baseline and DFC across selected domains, then analyze utility.",
    )
    parser.add_argument("--tau-extra-arg", action="append", default=[], help="Extra raw arg passed to tau2 run; repeat for each token.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_run:
        args.run_baseline = True
        args.run_dfc = True
        args.analyze = True
    repo_dir = args.repo_dir.resolve()
    policy_map = args.policy_map.resolve()
    names = RunNames(args.baseline_prefix, args.dfc_prefix)
    include_knowledge = "banking_knowledge" in args.domains

    ensure_uv()
    ensure_repo(repo_dir, args.repo_url, args.ref)
    if not args.skip_setup:
        uv_sync(repo_dir, include_knowledge=include_knowledge)
    if not args.skip_generate:
        generate_policy_map(repo_dir, args.domains, policy_map)
    install_adapter(repo_dir, policy_map)

    num_tasks = None if args.all_tasks else args.num_tasks
    if args.run_baseline:
        for domain in args.domains:
            run_tau(
                repo_dir,
                domain,
                agent="llm_agent",
                agent_model=args.agent_model,
                user_model=args.user_model,
                num_trials=args.num_trials,
                num_tasks=num_tasks,
                task_ids=args.task_ids,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                save_to=f"{names.baseline}_{domain}",
                retrieval_config=args.retrieval_config,
                extra_args=args.tau_extra_arg,
            )
    if args.run_dfc:
        for domain in args.domains:
            run_tau(
                repo_dir,
                domain,
                agent="dfc_llm_agent",
                agent_model=args.agent_model,
                user_model=args.user_model,
                num_trials=args.num_trials,
                num_tasks=num_tasks,
                task_ids=args.task_ids,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                save_to=f"{names.dfc}_{domain}",
                retrieval_config=args.retrieval_config,
                extra_args=args.tau_extra_arg,
            )
    if args.analyze:
        analyze(repo_dir, args.domains, names, args.analysis_dir.resolve())


if __name__ == "__main__":
    main()
