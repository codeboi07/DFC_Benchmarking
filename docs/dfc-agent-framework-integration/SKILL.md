# DFC Agent Framework Integration Guide

Reusable guide for integrating Passant data-flow-control (DFC) with agent benchmarks. Written after the AgentDyn (`agentdojo`) integration.

## Overview

The top-level `dfc_agent_framework_integration` package provides a benchmark-neutral lifecycle:

1. Create a single per-task DuckDB database and wrap it with DFC (`dfc(raw_conn)`).
2. Create empty event tables for every tool input/output relation, plus `PromptInput` and `AssistantResponseOutput`.
3. Extract trusted facts from the task preamble into a one-row `PreambleData` relation in that same database.
4. Generate and register DFC policies from the preamble + runtime schema.
5. During the run, validate writes into sink relations and accumulate allowed rows in the database.
6. Feed policy `DESCRIPTION` text back to the LLM on violations.
7. Close per-task DuckDB/DFC state before the next task.

AgentDyn-specific wiring lives in [`src/agentdojo/integrations/dfc_agent_framework_integration.py`](../../src/agentdojo/integrations/dfc_agent_framework_integration.py).

## Dependencies

- **`data-flow-control>=0.1.3`** (required). Version 0.1.3+ allows dimension subquery aliases in `CONSTRAINT` clauses (for example `AccountInfo.password` when `AccountInfo` is the alias on a `DIMENSION (SELECT ...) AccountInfo` line).
- Declared in [`dfc_agent_framework_integration/pyproject.toml`](../../dfc_agent_framework_integration/pyproject.toml).

For a local Passant checkout during development:

```bash
uv pip install -e ../oss-data-flow-control
```

## Run Database Model

Each task uses **one DuckDB connection** for everything — there is no secondary scratch database for policy probes or startup-only state.

| Relation kind | When rows are written | DFC-guarded? | Accumulates across run? |
|---------------|----------------------|--------------|-------------------------|
| `PreambleData` | Bootstrap (one row) | No | No (fixed at task start) |
| `*Input` | Before tool execution, via `validate_tool_call` / `validate_event` | Yes (staging → sink INSERT) | Yes |
| `*Output` | After successful tool execution, via `record_tool_output` | No (environment observation) | Yes |
| `PromptInput`, `AssistantResponseOutput` | Prompt/response validation | Yes | Yes |

Write path for guarded relations:

1. Stage the payload in `{Relation}WriteStaging`.
2. `INSERT INTO {Sink} SELECT ... FROM {Relation}WriteStaging` through `dfc_conn.execute(...)`.
3. DFC rewrites the INSERT and joins any `DIMENSION` / `SOURCE` relations referenced by registered policies.
4. Do **not** hard-code `CROSS JOIN PreambleData` in application SQL — Passant handles dimension joins.

Policy registration validates **parse + catalog only** (`Policy.from_pgn`, referenced tables exist in schema). Policies are **not** probed with synthetic preamble-derived payloads. Repair applies to parse or catalog errors; see [Repair vs delete](#repair-vs-delete) below.

## Separate DFC model vs agent model

Policy extraction, generation, and repair use a **different LLM** from the benchmark agent when configured:

```python
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

pipeline = AgentPipeline.from_config(
    PipelineConfig(
        llm="gpt-4o-2024-08-06",           # agent (tool-calling loop)
        dfc_model="gpt-5.2",                # extraction / generation / repair only
        defense="dfc_agent_framework_integration",
        suite_name="shopping",
        system_message_name=None,
        system_message="You are a helpful assistant.",
    )
)
```

- **`llm`**: drives the agent tool-calling loop.
- **`dfc_model`**: passed to `OpenAIStructuredLLMClient` for `responses.parse` during bootstrap. Defaults to the agent model when omitted.
- Exported **`metadata.json`** includes `dfc_model`, `agent_model`, and `model` (legacy alias for `dfc_model`).

In [`notebooks/shopping_dfc_end_to_end.ipynb`](../../notebooks/shopping_dfc_end_to_end.ipynb), set `MODEL` and `DFC_MODEL` separately in the config cell.

## Find the Benchmark Task Preamble

For AgentDyn / AgentDojo:

- **User tasks**: `BaseUserTask.PROMPT`
- **Injection utility checks**: `BaseInjectionTask.GOAL`

The pipeline receives the preamble as the `query` string in `AgentPipeline.query(...)`. The bootstrap uses `query` when no explicit `benchmark_task_context` is supplied in `extra_args`.

## Enumerate Tasks for Dry Preparation

```python
from agentdojo.integrations.dfc_agent_framework_integration import prepare_agentdyn_task_contexts

contexts = prepare_agentdyn_task_contexts(benchmark_version="v1.2.2")
```

This returns `BenchmarkTaskContext` objects for all `shopping`, `github`, and `dailylife` user tasks plus injection `GOAL` tasks without calling an LLM.

## Derive Tool / Input / Output Schemas

```python
from dfc_agent_framework_integration.schema import RuntimeSchema

runtime_schema = RuntimeSchema.from_tools(
    runtime.functions,
    functions_runtime=runtime,
    env=env,
)
```

Pass **`functions_runtime`** and **`env`** when building the schema at bootstrap time (AgentDyn does this in `AgentDojoDFCBootstrap`). This enables **dict-return output probing**: for tools whose return type is `dict[...]`, the integration calls the tool once with mock args against the suite environment. If the result is a flat scalar dict, columns are flattened (for example `"First Name"` → `first_name` on `GetShoppingAccountUserInformationOutput`). Nested dicts stay as `__dfc_raw_json`.

Relation naming convention (PascalCase):

- `send_email` → `SendEmailInput`, `SendEmailOutput`
- Final assistant text → `AssistantResponseOutput`
- Sub-agent prompts → `PromptInput`

Columns come from `Function.parameters.model_json_schema()["properties"]`. Parameter docstrings become `column_descriptions` on each tool input relation. List parameters also expose singular exploded columns (for example `recipients` → `recipient`) with derived descriptions. Tool output relations include return-type field schemas when the return type is a Pydantic model; probed flat dicts; or otherwise `__dfc_raw_json`.

## Live monitoring (`events.jsonl`)

Bootstrap and runtime validation write **incremental** progress logs under each task's `*_dfc` directory — you do not need to wait for `metadata.json` or `none.json` to finish.

```text
runs/<pipeline>/<suite>/<user_task>/<attack>/<stem>_dfc/
  events.jsonl     # append-only; flushed after each event
  status.json      # latest event snapshot (quick cat/jq)
  metadata.json    # full export at task end
  database.duckdb  # full DuckDB state at task end
```

Example:

```text
runs/gpt-4o-2024-08-06-dfc_agent_framework_integration/shopping/user_task_0/none/none_dfc/
```

Monitor while a run is in progress:

```bash
tail -f runs/.../user_task_2/none/none_dfc/events.jsonl
cat runs/.../none_dfc/status.json | jq .
```

| Event | Meaning |
|-------|---------|
| `prepare_task_start` | Bootstrap began |
| `extraction_start` / `extraction_complete` | Preamble fact extraction (`dfc_model`) |
| `extraction_repair_*` | Extraction validation failed; retrying |
| `policy_generation_start` / `policy_generation_llm_complete` | Policy LLM returned |
| `policy_register_start` / `policy_register_attempt_failed` | Registering one policy |
| `policy_repair_start` / `policy_repair_complete` | Repair LLM fixing a failed PGN |
| `policy_register_deleted` | Policy dropped after repair |
| `policy_generation_complete` / `prepare_task_complete` | Bootstrap done; agent loop next |
| `validation` (`blocked: true`) | Runtime DFC block (tool, prompt, or response) |
| `tool_output_recorded` | Successful tool result persisted to `*Output` |
| `context_closed` / `artifacts_exported` | Task teardown |

Implementation: [`dfc_event_log.py`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/dfc_event_log.py). The artifact directory is resolved from AgentDojo's active `TraceLogger` during bootstrap.

## Benchmark log files (`none.json` vs DFC artifacts)

| File | When it updates |
|------|-----------------|
| **`events.jsonl`** | Continuously during bootstrap and validation |
| **`none.json`** (AgentDojo task log) | At **pipeline element checkpoints** only — not during in-flight LLM calls |
| **`metadata.json`** | Once at task export (end of run) |

Implications:

- A `none.json` with only `system` + `user` messages means the run has **not yet completed the first agent LLM response** (often still in DFC bootstrap or waiting on the first `gpt-4o` call). It does **not** mean the task has not started — check `events.jsonl` instead.
- Once the agent is looping, `none.json` grows with `assistant` / `tool` messages after each checkpoint.
- `utility`, `security`, and `duration` appear in `none.json` only when the benchmark task completes.

## Runtime bounds (agent vs DFC)

The agent loop is **bounded**, not infinite — but each step can be slow (especially `dfc_model` bootstrap and OpenAI rate-limit backoff).

| Layer | Limit | Where |
|-------|-------|-------|
| Tool-calling rounds | **15** | `ToolsExecutionLoop(max_iters=15)` |
| Full pipeline restarts | **3** | `TaskSuite.run_task_with_pipeline` when no model output |
| Final response rewrites | **2** retries (3 attempts) | `AgentDojoDFCResponseValidator(max_retries=2)` |
| DFC tool blocks | Soft errors returned to agent | Retries count against the 15 tool rounds |
| OpenAI HTTP retries | Up to **8** attempts, backoff to **90s** | `openai_llm.chat_completion_request` |

DFC policy **repair** during bootstrap: up to **2** repair attempts per generated policy.

## Hook DFC Validation Into the Runtime Loop

### Bootstrap (before first LLM call)

- Call `DFCBenchmarkContext.prepare_task(...)` with optional `event_log=DFCEventLog(artifact_dir)`.
- Store the context in `extra_args["dfc_context"]`.
- Append a short system notice (not full policies).

### Tool calls

Replace or wrap `ToolsExecutor` to:

1. Call `context.validate_tool_call(tool_name, args)` **before** `runtime.run_function(...)`.
   - On violation, return a `ChatToolResultMessage` with `error` set to the policy description.
   - On success, the args are written into the corresponding `*Input` relation (DFC-guarded).
2. After a **successful** execution, call `context.record_tool_output(tool_name, result)`.
   - This appends a row to the `*Output` relation (not DFC-guarded).

Inputs are recorded at validation time, before the tool runs. If validation passes but execution fails, the input row remains in the database. Blocked validations never insert.

List-typed parameters (for example `recipients`) are exploded into one staged row per element during validation.

### Final responses

Add a post-loop validator that calls `context.validate_assistant_response(content)` and retries the LLM with a user message when blocked (up to `max_retries`).

### Prompts

`AgentDojoDFCPromptGuard` wraps the LLM and calls `context.validate_prompt(...)` when the latest message is a user turn. On violation, it appends policy feedback as a user message and then calls the LLM so the run still produces an assistant message. DFC-generated feedback messages are not re-validated.

### Generic events

```python
context.validate_event(
    event_type="prompt",
    relation="PromptInput",
    payload={"content": prompt_text, "target": target_name},
)
```

## Feed Policy Descriptions Back to the LLM

Generated policies must include a PGN `DESCRIPTION` clause. On violation, the wrapper returns concise feedback:

```
Data flow policy violation. The tool call was not executed.
Policy: <DESCRIPTION from registered policy>
Attempted tool: send_email
Attempted arguments: {...}
Try again using only values authorized by the original task.
```

Diagnostics also record validation events, policy fire counts, and registration outcomes in `DFCTaskDiagnostics`. On violation, policy IDs/descriptions are taken from registered policies that apply to the blocked relation.

## Guarantee Per-Task Cleanup

`DFCBenchmarkContext.close()` must run after every task:

```python
try:
    _, _, env, messages, extra_args = pipeline.query(...)
finally:
    context = extra_args.get("dfc_context")
    if context is not None:
        context.close()
```

AgentDyn implements this in `TaskSuite.run_task_with_pipeline` via `try/finally`.

## AgentDyn-Specific Choices (Do Not Copy Blindly)

| AgentDyn choice | Why it is specific |
|-----------------|-------------------|
| OpenAI `responses.parse` for structured outputs | Other benchmarks may use different providers; implement `StructuredLLMClient` |
| `--defense dfc_agent_framework_integration` pipeline shape | Defense ordering depends on how the benchmark loops tools vs final text |
| `RuntimeSchema.from_tools(..., functions_runtime, env)` | Other benchmarks may not have a sandbox env for dict output probing |
| PascalCase relation names from snake_case tools | Match your framework's naming or adopt the same convention consistently |
| `dfc_model` separate from agent `llm` | Optional; other benchmarks may use one model for everything |
| SINK vs SOURCE aggregates in `CONSTRAINT` | Reference SINK columns directly (no `max()`). Use aggregates such as `max()` only on SOURCE columns when the source relation may have multiple rows |

The old `--defense dfc` prototype (SQL `execute_intent` gateway, static `policies/shopping_policies.py`) remains for comparison. New integrations should use `dfc_agent_framework_integration`.

## Policy Generation Notes

LLM-facing policy patterns live in [`prompts.py`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/prompts.py) (`POLICY_EXAMPLES`, `STRUCTURED_OUTPUT_GUIDANCE`, `REPAIR_DECISION_GUIDANCE`). The SKILL summarizes operational rules integrators need.

### Structured output (PGN field pitfalls)

Common registration failures from malformed LLM JSON:

| Mistake | Result |
|---------|--------|
| `pgn` contains only `SINK SendEmailInput` | `missing required clause: CONSTRAINT` |
| Multiple `CONSTRAINT` lines in one policy | Parse error — combine with `AND` in **one** line |
| `DIMENSION AccountInfo (SELECT ...)` | Invalid dimension entry — alias goes **after** the subquery |
| `(SELECT ...) AccountInfo DIMENSION (SELECT ...)` | Duplicate dimension — one subquery, one alias |
| Moving clauses into `description` / `rationale` | Incomplete `pgn`; repair or delete |

Valid shape — full multi-line string in `pgn`:

```text
SINK SendEmailInput
DIMENSION PreambleData
CONSTRAINT SendEmailInput.recipient = PreambleData.authorized_recipient_email
ON FAIL KILL
DESCRIPTION Prevent data exfiltration: outbound email may go only to the recipient authorized by the task.
```

Combine multiple checks in one constraint:

```text
CONSTRAINT (AccountInfo.email = '' OR NOT contains(SendEmailInput.body, AccountInfo.email)) AND (AccountInfo.password = '' OR NOT contains(SendEmailInput.body, AccountInfo.password))
```

### Exfiltration and defensive policies

- **Pre-emptive guards** on external sinks (`SendEmailInput`, `BrowseWebpageInput`, `InputToWebpageInput`, `SendMoneyInput`) are intentional even when the trusted task does not mention email, web forms, or money. The agent may still call those tools due to injection or drift.
- Prefer **`DIMENSION`** (not `SOURCE REQUIRED`) on sensitive `*Output` relations for exfiltration checks — applies when private data is present without forcing an earlier read.
- When an `*Output` relation may be **empty** (read tool not called yet), use a **dimension subquery** so the dimension always contributes one row:

```text
DIMENSION (SELECT COALESCE(max(password), '') AS password FROM GetShoppingAccountUserInformationOutput) AccountInfo
CONSTRAINT AccountInfo.password = '' OR NOT contains(SendEmailInput.body, AccountInfo.password)
ON FAIL KILL
```

- Use **`contains()`** for substring exfiltration (password embedded in a longer email body). Avoid `ILIKE`/`LIKE` — they fail at runtime. Exact equality is insufficient for body exfiltration checks.
- With **`data-flow-control>=0.1.3`**, reference subquery dimension aliases in constraints (`AccountInfo.password`).

### General PGN rules

- Almost every generated policy should declare a `SINK`; validation runs on writes into sink relations.
- `DIMENSION` may reference **any relation in the schema**: `PreambleData`, tool `*Output` rows accumulated during the run, or other input relations.
- Use `SOURCE` / `SOURCE REQUIRED` only when governing reads from existing relations.
- Reference **SINK** and **DIMENSION** columns directly — no aggregates on those. **SOURCE** columns may need `max()` when the source has multiple rows.
- List tool parameters are exploded into singular columns (for example `recipients` → `recipient`).
- Registration checks PGN parse + that referenced tables/columns exist. It does **not** execute probe writes at registration time.
- Always use `ON FAIL KILL`. Do not use `ON FAIL REMOVE`, `ON FAIL LLM`, or unsupported aggregate policy syntax.

### Repair vs delete

When policy registration fails, the repair LLM should:

- **Default to repair** — return a complete `repaired_pgn`.
- **Not delete** because the task does not use the sink's tool or lacks preamble grounding for exfiltration policies.
- **Delete only** when the policy is unrecoverable (references relations/columns that do not exist and cannot be rewritten).

See `REPAIR_DECISION_GUIDANCE` in [`prompts.py`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/prompts.py).

### Policy examples (aligned with `prompts.py`)

| # | Pattern |
|---|---------|
| 1 | Email recipient grounded to `PreambleData` |
| 2 | URL grounded to `PreambleData` |
| 3 | Money transfer grounded to `PreambleData` |
| 4 | Exfiltration — email recipient allowlist |
| 5 | Exfiltration — password in body via subquery dimension + `contains()` |
| 6 | Exfiltration — web form URL allowlist |

Example 5 (exfiltration with empty-output-safe dimension):

```text
SINK SendEmailInput
DIMENSION (SELECT COALESCE(max(password), '') AS password FROM GetShoppingAccountUserInformationOutput) AccountInfo
CONSTRAINT AccountInfo.password = '' OR NOT contains(SendEmailInput.body, AccountInfo.password)
ON FAIL KILL
DESCRIPTION Prevent data exfiltration: do not email the account password retrieved during this run.
```

Plain table dimensions (for example `DIMENSION SearchProductOutput`) require rows to already exist from an earlier tool call in the same run.

## Notebook and full-suite benchmark ops

[`notebooks/shopping_dfc_end_to_end.ipynb`](../../notebooks/shopping_dfc_end_to_end.ipynb) runs shopping benchmarks with configurable defenses:

- Set **`MODEL`** (agent) and **`DFC_MODEL`** (bootstrap) separately.
- **`WORKERS`**: thread-pool concurrency. `4` is reasonable; reduce to `1` when debugging rate limits or hangs.
- **`QUICK_TEST = True`**: one user task + one injection — use for smoke runs.
- **`FORCE_RERUN = False`**: skips tasks that already have valid logs under `runs/`.

**Job count vs work:** the notebook tqdm shows **40 jobs** (20 clean + 20 attack), but each **`attack` job runs 9 injection tasks sequentially** inside one thread-pool slot. One slow attack job can block a tqdm tick for many minutes (9 × bootstrap + agent run).

Rate limits: OpenAI calls use exponential backoff (up to 90s × 8 attempts). Parallel workers share one pipeline/OpenAI client — concurrent `dfc_model` bootstrap + agent calls can queue or 429 together. Use `events.jsonl` to see whether time is spent in extraction, policy repair, or runtime `validation` events.

## Testing Checklist

- [ ] Fake `StructuredLLMClient` covers extraction, policy generation, and repair paths
- [ ] `PreambleData` materializes as exactly one row with validated snake_case keys in the task DuckDB
- [ ] Invalid PGN triggers repair; `delete=True` drops only the failed policy when unrecoverable
- [ ] Sink-only or multi-`CONSTRAINT` PGN triggers repair to a complete single-constraint policy
- [ ] `contains()` exfiltration policy registers and blocks substring match in email body
- [ ] Dimension subquery + alias constraint registers with `data-flow-control>=0.1.3`
- [ ] Blocked tool calls do not execute; allowed calls execute once
- [ ] Allowed tool inputs appear in `*Input` relations; successful outputs appear in `*Output` relations
- [ ] Cross-step policies can use `*Output` relations as `DIMENSION` after prior tools run
- [ ] Violation messages include policy `DESCRIPTION` text
- [ ] `events.jsonl` written during `prepare_task` (not only at final export)
- [ ] `dfc_model` wired separately from agent model when using `PipelineConfig.dfc_model`
- [ ] `close()` runs on success and exception
- [ ] Dry enumeration of all task preambles matches suite `PROMPT` / `GOAL` exactly

## Smoke Run (opt-in, requires API keys)

Ensure `data-flow-control>=0.1.3` is installed (`uv sync` from repo root).

```bash
uv pip install -e ../oss-data-flow-control   # optional: local Passant checkout
uv run python -m agentdojo.scripts.benchmark \
  -s shopping \
  --model GPT_4O_MINI_2024_07_18 \
  --defense dfc_agent_framework_integration \
  --attack important_instructions \
  --user-task user_task_1 \
  --injection-task injection_task_0 \
  --force-rerun
```

Run one task per suite (`shopping`, `github`, `dailylife`) before full-suite benchmarks. Watch bootstrap progress:

```bash
tail -f runs/gpt-4o-mini-2024-07-18-dfc_agent_framework_integration/shopping/user_task_1/none/none_dfc/events.jsonl
```

## Key Files

- Library: [`dfc_agent_framework_integration/src/dfc_agent_framework_integration/`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/)
- Live event log: [`dfc_event_log.py`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/dfc_event_log.py)
- Policy prompts / examples: [`prompts.py`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/prompts.py)
- AgentDyn adapter: [`src/agentdojo/integrations/dfc_agent_framework_integration.py`](../../src/agentdojo/integrations/dfc_agent_framework_integration.py)
- Pipeline registration: [`src/agentdojo/agent_pipeline/agent_pipeline.py`](../../src/agentdojo/agent_pipeline/agent_pipeline.py) (`dfc_agent_framework_integration` defense, `PipelineConfig.dfc_model`)
- Task cleanup: [`src/agentdojo/task_suite/task_suite.py`](../../src/agentdojo/task_suite/task_suite.py)
- End-to-end notebook: [`notebooks/shopping_dfc_end_to_end.ipynb`](../../notebooks/shopping_dfc_end_to_end.ipynb)
- Tests: [`dfc_agent_framework_integration/tests/`](../../dfc_agent_framework_integration/tests/), [`tests/test_agent_pipeline/test_dfc_agent_framework_integration.py`](../../tests/test_agent_pipeline/test_dfc_agent_framework_integration.py)
