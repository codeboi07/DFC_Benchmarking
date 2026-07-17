#!/usr/bin/env python3
"""
Spider2-DBT Sales task model-comparison runner.

This is intentionally lightweight and Colab-friendly:
  - Uses AWS Bedrock Runtime Converse API.
  - Runs a ReAct-style DBT agent with Harbor-like trajectory logging.
  - Copies each task into an isolated run workspace.
  - Saves full tool calls, tool results, final files, dbt logs, and comparison CSVs.

Expected task source layout:
  <task_root>/<instance_id>/
    dbt_project.yml
    profiles.yml
    models/
    *.duckdb

For Spider2-DBT, <task_root> is usually the extracted DBT_start_db folder.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3


DEFAULT_BIG_MODEL_ID = "anthropic.claude-opus-4-8"
DEFAULT_SMALL_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

SALES_TASKS = [
    "asana001",
    "workday002",
    "scd001",
    "lever001",
    "greenhouse001",
    "qualtrics001",
    "salesforce001",
    "hubspot001",
    "jira001",
    "marketo001",
    "workday001",
    "twilio001",
    "intercom001",
]

SYSTEM_MESSAGE = """\
You are a data scientist proficient in databases, SQL, and DBT projects.
You are starting in the /workspace directory, which contains all the codebase needed for your task.
You can only use the actions provided in the ACTION SPACE to solve the task.
For each step, output exactly one Thought and one Action. The Action cannot be empty.
The maximum number of steps is {max_steps}.

# ACTION SPACE

## Bash Action
Signature: Bash(code="shell_command")
Description: Execute a valid non-interactive shell command in the current task workspace.
Examples:
Action: Bash(code="ls -la")
Action: Bash(code="dbt run --profiles-dir .")

## LOCAL_DB_SQL Action
Signature: LOCAL_DB_SQL(file_path="database.duckdb", command="sql_command", output="direct" or "path/to/output.csv")
Description: Execute SQL against a DuckDB database. If output is direct, show the rows. If output is a path, save CSV.
Example:
Action: LOCAL_DB_SQL(file_path="salesforce.duckdb", command="select * from my_table limit 10", output="direct")

## CreateFile
Signature: CreateFile(filepath="path/to/file"):
```file_content```
Description: Create a new file. If the file exists, use EditFile.

## EditFile
Signature: EditFile(filepath="path/to/file"):
```file_content```
Description: Overwrite an existing file. Normally read the file first.

## Terminate Action
Signature: Terminate(output="literal_answer_or_output_path")
Description: Complete the task. For DBT tasks, return the DuckDB filename, not a CSV.

# DBT PROJECT RULES

1. First inspect dbt_project.yml, profiles.yml, models, YAML files, docs, and relevant SQL files.
2. The project is unfinished. Use YAML definitions to identify missing or incomplete model SQL files.
3. Do not modify YAML unless absolutely required. Prefer creating/fixing SQL models.
4. If packages.yml exists, run dbt deps before dbt run.
5. Run dbt run --profiles-dir . to execute the transformations.
6. Verify generated models where practical using LOCAL_DB_SQL.
7. Do not terminate until all required SQL models are complete according to YAML and dbt succeeds.
8. Do not use networking, privilege escalation, destructive system commands, or interactive editors.

# RESPONSE FORMAT

Thought: concise reasoning for the next step.
Action: one valid action from the action space.
"""

TASK_TEMPLATE = """\
# {instance_id}

## Task Description

{instruction}

## Environment

You are working with a dbt project that transforms data in a DuckDB database.
The project is located in `/workspace` with this typical structure:

- `dbt_project.yml` - dbt project configuration
- `profiles.yml` - DuckDB profile
- `models/` - SQL transformation models
- `*.duckdb` - Source database containing raw data and generated model outputs

## Objective

Complete or fix the dbt models to produce the correct table outputs.
Run `dbt run --profiles-dir .` to execute the transformations.

Safety note: operate only inside the local task sandbox. Do not attempt privilege escalation,
networking, or destructive actions.
"""


@dataclass
class ModelSpec:
    label: str
    model_id: str
    provider: str = "bedrock"
    temperature: float = 0.0
    max_tokens: int = 4096


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_jsonl_manifest(manifest: Optional[Path]) -> Dict[str, str]:
    if not manifest or not manifest.exists():
        return {}
    out: Dict[str, str] = {}
    text = manifest.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    # Handles both normal JSONL and Spider2's occasionally space-separated JSON objects.
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        if isinstance(row, dict) and "instance_id" in row:
            out[row["instance_id"]] = row.get("instruction", "")
    return out


def discover_instruction(task_dir: Path, instance_id: str, manifest_map: Dict[str, str]) -> str:
    if instance_id in manifest_map:
        return manifest_map[instance_id]
    # Fallback: use README/markdown if present.
    for name in ("README.md", f"{instance_id}.md"):
        p = task_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")[:6000]
    return f"Complete the DBT transformation project for {instance_id}."


def copy_task_workspace(task_root: Path, instance_id: str, dest: Path) -> None:
    src = task_root / instance_id
    if not src.exists():
        raise FileNotFoundError(f"Task directory not found: {src}")
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("target", "logs", ".git", "__pycache__")
    shutil.copytree(src, dest, ignore=ignore)


def safe_path(workspace: Path, user_path: str) -> Path:
    p = Path(user_path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (workspace / p).resolve()
    ws = workspace.resolve()
    if ws != resolved and ws not in resolved.parents:
        raise ValueError(f"Path escapes workspace: {user_path}")
    return resolved


def truncate_for_observation(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated for model context; full output saved in trajectory, {len(text)} chars]"


def parse_action(response: str) -> Tuple[str, Dict[str, Any]]:
    m = re.search(r"Action:\s*(.*)", response, flags=re.DOTALL)
    action_text = (m.group(1).strip() if m else response.strip())
    action_text = action_text.strip()
    if action_text.startswith("```") and action_text.endswith("```"):
        action_text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", action_text)
        action_text = re.sub(r"\n?```$", "", action_text).strip()

    term = re.search(r"""Terminate\s*\(\s*output\s*=\s*(['"])(.*?)\1\s*\)""", action_text, re.DOTALL)
    if term:
        return "Terminate", {"output": term.group(2), "raw": action_text}

    bash = re.search(r"""Bash\s*\(\s*code\s*=\s*(['"])((?:\\.|(?!\1).)*)\1\s*\)""", action_text, re.DOTALL)
    if bash:
        return "Bash", {"code": bytes(bash.group(2), "utf-8").decode("unicode_escape"), "raw": action_text}

    sql = re.search(
        r"""LOCAL_DB_SQL\s*\(\s*file_path\s*=\s*(['"])(.*?)\1\s*,\s*command\s*=\s*(['"])((?:\\.|(?!\3).)*)\3\s*,\s*output\s*=\s*(['"])(.*?)\5\s*\)""",
        action_text,
        re.DOTALL,
    )
    if sql:
        return "LOCAL_DB_SQL", {
            "file_path": sql.group(2),
            "command": bytes(sql.group(4), "utf-8").decode("unicode_escape"),
            "output": sql.group(6),
            "raw": action_text,
        }

    cf = re.search(r"""(CreateFile|EditFile)\s*\(\s*filepath\s*=\s*(['"])(.*?)\2\s*\)\s*:\s*```(?:[a-zA-Z0-9_+-]*)?\n?(.*?)```""", action_text, re.DOTALL)
    if cf:
        return cf.group(1), {"filepath": cf.group(3), "content": cf.group(4), "raw": action_text}

    raise ValueError(f"Could not parse action from response:\n{response[:1000]}")


class BedrockChat:
    def __init__(self, model: ModelSpec, region: str):
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        br_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ]
        kwargs: Dict[str, Any] = {
            "modelId": self.model.model_id,
            "messages": br_messages,
            "system": [{"text": system}],
            "inferenceConfig": {
                "maxTokens": self.model.max_tokens,
                "temperature": self.model.temperature,
            },
        }
        response = self.client.converse(**kwargs)
        text_parts = []
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
        return {
            "text": "\n".join(text_parts).strip(),
            "raw_response_metadata": response.get("ResponseMetadata", {}),
            "usage": response.get("usage", {}),
            "metrics": response.get("metrics", {}),
        }


def execute_action(action_type: str, args: Dict[str, Any], workspace: Path, timeout: int) -> Dict[str, Any]:
    started = time.time()
    if action_type == "Bash":
        proc = subprocess.run(
            args["code"],
            shell=True,
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "kind": "Bash",
            "command": args["code"],
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - started, 3),
        }

    if action_type in ("CreateFile", "EditFile"):
        path = safe_path(workspace, args["filepath"])
        exists = path.exists()
        if action_type == "CreateFile" and exists:
            raise FileExistsError(f"CreateFile target already exists: {args['filepath']}")
        if action_type == "EditFile" and not exists:
            raise FileNotFoundError(f"EditFile target does not exist: {args['filepath']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        before_hash = sha256_file(path) if exists else None
        path.write_text(args["content"], encoding="utf-8")
        after_hash = sha256_file(path)
        return {
            "kind": action_type,
            "path": str(path.relative_to(workspace)),
            "before_hash": before_hash,
            "after_hash": after_hash,
            "stdout": f"{action_type} completed for {args['filepath']}",
            "stderr": "",
            "exit_code": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }

    if action_type == "LOCAL_DB_SQL":
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("duckdb package is required for LOCAL_DB_SQL") from exc
        db_path = safe_path(workspace, args["file_path"])
        output = args["output"]
        con = duckdb.connect(str(db_path))
        try:
            result = con.execute(args["command"])
            if output.lower() in ("direct", "directly"):
                rows = result.fetchmany(100)
                cols = [d[0] for d in result.description] if result.description else []
                stdout = json.dumps({"columns": cols, "rows": rows}, default=str, indent=2)
            else:
                out_path = safe_path(workspace, output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Re-run through COPY to avoid pandas dependency for this action.
                escaped = args["command"].replace("'", "''")
                con.execute(f"COPY ({escaped}) TO '{str(out_path)}' (HEADER, DELIMITER ',')")
                stdout = f"Saved SQL output to {output}"
        finally:
            con.close()
        return {
            "kind": "LOCAL_DB_SQL",
            "file_path": args["file_path"],
            "command": args["command"],
            "output": output,
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "elapsed_sec": round(time.time() - started, 3),
        }

    if action_type == "Terminate":
        return {
            "kind": "Terminate",
            "output": args["output"],
            "exit_code": 0,
            "stdout": f"Terminated with output: {args['output']}",
            "stderr": "",
            "elapsed_sec": round(time.time() - started, 3),
        }

    raise ValueError(f"Unsupported action type: {action_type}")


def snapshot_files(workspace: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in workspace.rglob("*"):
        if p.is_file():
            rel = p.relative_to(workspace).as_posix()
            if rel.startswith(("target/", "logs/", "dbt_packages/")):
                continue
            try:
                out[rel] = sha256_file(p)
            except OSError:
                pass
    return out


def diff_snapshots(before: Dict[str, str], after: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            op = "created" if path not in before else "deleted" if path not in after else "modified"
            rows.append({
                "path": path,
                "operation": op,
                "before_hash": before.get(path),
                "after_hash": after.get(path),
            })
    return rows


def build_observation(tool_result: Dict[str, Any]) -> str:
    if tool_result["kind"] == "Terminate":
        return tool_result["stdout"]
    parts = []
    if "exit_code" in tool_result:
        parts.append(f"exit_code={tool_result['exit_code']}")
    stdout = tool_result.get("stdout") or ""
    stderr = tool_result.get("stderr") or ""
    if stdout:
        parts.append("STDOUT:\n" + stdout)
    if stderr:
        parts.append("STDERR:\n" + stderr)
    return truncate_for_observation("\n".join(parts).strip() or "(no output)")


def classify_run(trajectory: List[Dict[str, Any]], finished: bool) -> Dict[str, Any]:
    actions = [s.get("action_type") for s in trajectory]
    bash_cmds = [s.get("tool_call", {}).get("arguments", {}).get("code", "") for s in trajectory if s.get("action_type") == "Bash"]
    dbt_runs = [s for s in trajectory if s.get("action_type") == "Bash" and "dbt run" in s.get("tool_call", {}).get("arguments", {}).get("code", "")]
    last_dbt = dbt_runs[-1] if dbt_runs else None
    dbt_success = bool(last_dbt and last_dbt.get("tool_result", {}).get("exit_code") == 0)
    file_edits = sum(1 for a in actions if a in ("CreateFile", "EditFile"))
    sql_queries = sum(1 for a in actions if a == "LOCAL_DB_SQL")
    inspected_yaml = any((".yml" in c or ".yaml" in c) and any(x in c for x in ("cat", "sed", "grep", "find", "ls", "rg")) for c in bash_cmds)
    inspected_models = any("models" in c for c in bash_cmds)
    terminated = bool(actions and actions[-1] == "Terminate")
    error_type = None
    if not finished:
        error_type = "max_steps_or_exception"
    elif not dbt_runs:
        error_type = "no_dbt_run"
    elif not dbt_success:
        error_type = "dbt_failure"
    elif not file_edits:
        error_type = "no_file_changes"
    elif sql_queries == 0:
        error_type = "no_output_verification"
    return {
        "finished": finished,
        "terminated": terminated,
        "steps": len(trajectory),
        "dbt_run_count": len(dbt_runs),
        "dbt_run_success": dbt_success,
        "file_edit_count": file_edits,
        "sql_query_count": sql_queries,
        "inspected_yaml": inspected_yaml,
        "inspected_models": inspected_models,
        "error_type": error_type,
    }


def run_one_task(
    *,
    task_root: Path,
    manifest_map: Dict[str, str],
    out_root: Path,
    instance_id: str,
    model: ModelSpec,
    region: str,
    trial: int,
    max_steps: int,
    command_timeout: int,
) -> Path:
    run_id = f"{dt.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}__{model.label}__trial{trial}__{instance_id}"
    run_dir = out_root / instance_id / model.label / f"trial_{trial}"
    workspace = run_dir / "workspace"
    agent_dir = run_dir / "agent"
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    for d in (agent_dir, logs_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    copy_task_workspace(task_root, instance_id, workspace)
    before = snapshot_files(workspace)
    instruction = discover_instruction(task_root / instance_id, instance_id, manifest_map)
    task_text = TASK_TEMPLATE.format(instance_id=instance_id, instruction=instruction)
    system_text = SYSTEM_MESSAGE.format(max_steps=max_steps)
    chat = BedrockChat(model, region)

    messages = [{"role": "user", "content": task_text + "\n\nObservation: You are in the folder now."}]
    trajectory: List[Dict[str, Any]] = []
    finished = False
    result: Optional[str] = None
    exception: Optional[str] = None

    for step in range(1, max_steps + 1):
        step_started = utc_now()
        try:
            completion = chat.complete(system_text, messages)
            response_text = completion["text"]
            action_type, action_args = parse_action(response_text)
            tool_result = execute_action(action_type, action_args, workspace, command_timeout)
            obs = build_observation(tool_result)

            tool_call = {"tool": action_type, "arguments": {k: v for k, v in action_args.items() if k != "raw"}}
            entry = {
                "step": step,
                "timestamp": step_started,
                "observation": messages[-1]["content"].split("Observation:", 1)[-1].strip(),
                "response": response_text,
                "thought": extract_thought(response_text),
                "action": action_args.get("raw", ""),
                "action_type": action_type,
                "tool_call": tool_call,
                "tool_result": tool_result,
                "model_usage": completion.get("usage", {}),
                "model_metrics": completion.get("metrics", {}),
            }
            trajectory.append(entry)
            (logs_dir / "stdout.log").open("a", encoding="utf-8").write(
                f"\n\n--- step {step} {action_type} ---\n{tool_result.get('stdout','')}"
            )
            (logs_dir / "stderr.log").open("a", encoding="utf-8").write(
                f"\n\n--- step {step} {action_type} ---\n{tool_result.get('stderr','')}"
            )

            if action_type == "Bash" and "dbt run" in action_args.get("code", ""):
                (logs_dir / "dbt_run.log").write_text(
                    (tool_result.get("stdout") or "") + "\n" + (tool_result.get("stderr") or ""),
                    encoding="utf-8",
                )

            if action_type == "Terminate":
                finished = True
                result = action_args["output"]
                break

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "Observation: " + obs})

        except ValueError as exc:
            # Parsing errors are common in long agent runs. Keep the run alive and ask for a corrected action.
            exception = repr(exc)
            trajectory.append({
                "step": step,
                "timestamp": step_started,
                "observation": messages[-1]["content"].split("Observation:", 1)[-1].strip(),
                "response": locals().get("response_text", ""),
                "thought": extract_thought(locals().get("response_text", "")),
                "action": "",
                "action_type": "PARSE_ERROR",
                "tool_call": {},
                "tool_result": {"kind": "PARSE_ERROR", "stderr": exception, "exit_code": 1},
            })
            messages.append({"role": "assistant", "content": locals().get("response_text", "")})
            messages.append({
                "role": "user",
                "content": (
                    "Observation: Your last response could not be parsed as a valid action. "
                    "Please output exactly one Thought and one Action using the provided action signatures. "
                    f"Parse error: {exception}"
                ),
            })
            continue
        except Exception as exc:  # Log and stop this run; the trace is still useful.
            exception = repr(exc)
            trajectory.append({
                "step": step,
                "timestamp": step_started,
                "observation": messages[-1]["content"].split("Observation:", 1)[-1].strip(),
                "response": "",
                "thought": "",
                "action": "",
                "action_type": "EXCEPTION",
                "tool_call": {},
                "tool_result": {"kind": "EXCEPTION", "stderr": exception, "exit_code": 1},
            })
            break

    after = snapshot_files(workspace)
    files_changed = diff_snapshots(before, after)
    evaluation = classify_run(trajectory, finished)
    if exception:
        evaluation["exception"] = exception

    # Copy dbt artifacts if produced.
    for artifact in ("manifest.json", "run_results.json"):
        src = workspace / "target" / artifact
        if src.exists():
            shutil.copy2(src, artifacts_dir / artifact)

    traj_obj = {
        "run_id": run_id,
        "benchmark": "spider2-dbt",
        "instance_id": instance_id,
        "family": "Sales",
        "trial": trial,
        "model": asdict(model),
        "region": region,
        "Task": task_text,
        "system_message": system_text,
        "trajectory": trajectory,
        "files_changed": files_changed,
        "finished": finished,
        "result": result,
        "evaluation": evaluation,
        "created_at": utc_now(),
    }
    traj_path = agent_dir / "spider-agent-dbt.trajectory.json"
    traj_path.write_text(json.dumps(traj_obj, indent=2, default=str), encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return run_dir


def extract_thought(response: str) -> str:
    m = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", response, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def load_eval(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(out_root: Path, output_csv: Path) -> None:
    rows = []
    for eval_path in out_root.glob("*/*/trial_*/evaluation.json"):
        run_dir = eval_path.parent
        instance_id = run_dir.parent.parent.name
        model_label = run_dir.parent.name
        trial = run_dir.name.replace("trial_", "")
        ev = load_eval(eval_path)
        row = {"instance_id": instance_id, "model": model_label, "trial": trial, **ev}
        rows.append(row)
    if not rows:
        print(f"No evaluation files found under {out_root}")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")

    # Pairwise wide comparison for small vs big.
    by_task_trial: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_task_trial.setdefault((r["instance_id"], r["trial"]), {})[r["model"]] = r
    compare_path = output_csv.with_name(output_csv.stem + "_paired.csv")
    compare_rows = []
    for (instance_id, trial), models in sorted(by_task_trial.items()):
        if "small_qwen235b" not in models or "big_opus48" not in models:
            continue
        s = models["small_qwen235b"]
        b = models["big_opus48"]
        compare_rows.append({
            "instance_id": instance_id,
            "trial": trial,
            "small_success": s.get("dbt_run_success"),
            "big_success": b.get("dbt_run_success"),
            "small_steps": s.get("steps"),
            "big_steps": b.get("steps"),
            "small_file_edits": s.get("file_edit_count"),
            "big_file_edits": b.get("file_edit_count"),
            "small_sql_queries": s.get("sql_query_count"),
            "big_sql_queries": b.get("sql_query_count"),
            "small_error_type": s.get("error_type"),
            "big_error_type": b.get("error_type"),
        })
    if compare_rows:
        with compare_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compare_rows)
        print(f"Wrote {compare_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-root", type=Path, required=True, help="Path to extracted Spider2-DBT task dirs, e.g. DBT_start_db")
    p.add_argument("--manifest", type=Path, default=None, help="Optional spider2-dbt.jsonl manifest with instructions")
    p.add_argument("--out-root", type=Path, default=Path("runs/spider2-dbt-sales"))
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--command-timeout", type=int, default=120)
    p.add_argument("--tasks", nargs="*", default=SALES_TASKS, help="Task ids to run; default is all 13 sales tasks")
    p.add_argument("--models", nargs="*", default=["small", "big"], choices=["small", "big"])
    p.add_argument("--small-model-id", default=DEFAULT_SMALL_MODEL_ID)
    p.add_argument("--big-model-id", default=DEFAULT_BIG_MODEL_ID)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args.out_root, args.out_root / "sales_comparison.csv")
        return

    manifest_map = read_jsonl_manifest(args.manifest)
    model_specs = []
    if "small" in args.models:
        model_specs.append(ModelSpec(label="small_qwen235b", model_id=args.small_model_id, max_tokens=4096))
    if "big" in args.models:
        model_specs.append(ModelSpec(label="big_opus48", model_id=args.big_model_id, max_tokens=4096))

    for trial in range(1, args.trials + 1):
        for instance_id in args.tasks:
            for model in model_specs:
                print(f"\n=== Running {instance_id} | {model.label} | trial {trial} ===", flush=True)
                run_dir = run_one_task(
                    task_root=args.task_root,
                    manifest_map=manifest_map,
                    out_root=args.out_root,
                    instance_id=instance_id,
                    model=model,
                    region=args.region,
                    trial=trial,
                    max_steps=args.max_steps,
                    command_timeout=args.command_timeout,
                )
                print(f"Saved run: {run_dir}", flush=True)

    aggregate(args.out_root, args.out_root / "sales_comparison.csv")


if __name__ == "__main__":
    main()
