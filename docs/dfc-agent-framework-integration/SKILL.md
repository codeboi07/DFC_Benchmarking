# DFC Agent Framework Integration Guide

Reusable guide for integrating Passant data-flow-control (DFC) with agent benchmarks. Written after the AgentDyn (`agentdojo`) integration.

## Overview

The top-level `dfc_agent_framework_integration` package provides a benchmark-neutral lifecycle:

1. Extract trusted facts from the task preamble into a one-row `PreambleData` relation.
2. Generate and register DFC policies from the preamble + runtime schema.
3. Validate tool calls, prompts, and assistant responses at runtime.
4. Feed policy `DESCRIPTION` text back to the LLM on violations.
5. Close per-task DuckDB/DFC state before the next task.

AgentDyn-specific wiring lives in [`src/agentdojo/integrations/dfc_agent_framework_integration.py`](../../src/agentdojo/integrations/dfc_agent_framework_integration.py).

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

runtime_schema = RuntimeSchema.from_tools(runtime.functions)
```

Relation naming convention (PascalCase):

- `send_email` → `SendEmailInput`, `SendEmailOutput`
- Final assistant text → `AssistantResponseOutput`
- Sub-agent prompts → `PromptInput`

Columns come from `Function.parameters.model_json_schema()["properties"]`. Parameter docstrings become `column_descriptions` on each tool input relation. List parameters also expose singular exploded columns (for example `recipients` → `recipient`) with derived descriptions. Tool output relations include return-type field schemas when the return type is a Pydantic model; otherwise a `__dfc_raw_json` column documents the serialized tool result.

## Hook DFC Validation Into the Runtime Loop

### Bootstrap (before first LLM call)

- Call `DFCBenchmarkContext.prepare_task(...)`.
- Store the context in `extra_args["dfc_context"]`.
- Append a short system notice (not full policies).

### Tool calls

Replace or wrap `ToolsExecutor` to call `context.validate_tool_call(tool_name, args)` before `runtime.run_function(...)`. On violation, return a `ChatToolResultMessage` with `error` set to the policy description.

### Final responses

Add a post-loop validator that calls `context.validate_assistant_response(content)` and retries the LLM with a user message when blocked.

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

Diagnostics also record grouped and replayed policy matches in `DFCTaskDiagnostics`.

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
| `RuntimeSchema.from_tools(agentdojo.Function)` | Other benchmarks may not use Pydantic tool schemas |
| PascalCase relation names from snake_case tools | Match your framework's naming or adopt the same convention consistently |
| Source aggregates in DIMENSION constraints (`max(...)`) | Required by Passant when validating staged rows via `SELECT` |

The old `--defense dfc` prototype (SQL `execute_intent` gateway, static `policies/shopping_policies.py`) remains for comparison. New integrations should use `dfc_agent_framework_integration`.

## Policy Generation Notes

- Use `SINK <ToolInput> DIMENSION PreambleData` for normal tool-call validation.
- Almost every generated policy should declare a `SINK`; validation runs on writes into sink relations.
- Use `SOURCE` / `SOURCE REQUIRED` only when governing reads from existing relations (for example `Receipts -> Expenses`).
- Sink columns in `CONSTRAINT` expressions must use aggregates such as `max()` when paired with `DIMENSION` tables.
- List tool parameters are exploded into singular columns (for example `recipients` → `recipient`) with one staged row per element.
- Avoid `ILIKE`/`LIKE`/`strpos` with aggregates; they fail at runtime with DuckDB binder errors. Prefer exact equality.
- Policies are runtime-probed on a scratch DuckDB connection before registration; probe failures trigger repair/delete.
- Always use `ON FAIL KILL`. Do not use `ON FAIL REMOVE`.
- `ON FAIL KILL` aborts the query with an explicit policy-violation signal instead of silently filtering rows.
- Do not generate `ON FAIL LLM`, invalidation columns, or aggregate policy syntax unsupported by Passant.
- Final assistant responses are fail-closed: after `max_retries`, violating text is replaced with a blocked message.

## Testing Checklist

- [ ] Fake `StructuredLLMClient` covers extraction, policy generation, and repair paths
- [ ] `PreambleData` materializes as exactly one row with validated snake_case keys
- [ ] Invalid PGN triggers repair; `delete=True` drops only the failed policy
- [ ] Blocked tool calls do not execute; allowed calls execute once
- [ ] Violation messages include policy `DESCRIPTION` text
- [ ] `close()` runs on success and exception
- [ ] Dry enumeration of all task preambles matches suite `PROMPT` / `GOAL` exactly

## Smoke Run (opt-in, requires API keys)

```bash
uv pip install -e ../oss-data-flow-control   # local Passant checkout if needed
uv run python -m agentdojo.scripts.benchmark \
  -s shopping \
  --model GPT_4O_MINI_2024_07_18 \
  --defense dfc_agent_framework_integration \
  --attack important_instructions \
  --user-task user_task_1 \
  --injection-task injection_task_0 \
  --force-rerun
```

Run one task per suite (`shopping`, `github`, `dailylife`) before full-suite benchmarks.

## Key Files

- Library: [`dfc_agent_framework_integration/src/dfc_agent_framework_integration/`](../../dfc_agent_framework_integration/src/dfc_agent_framework_integration/)
- AgentDyn adapter: [`src/agentdojo/integrations/dfc_agent_framework_integration.py`](../../src/agentdojo/integrations/dfc_agent_framework_integration.py)
- Pipeline registration: [`src/agentdojo/agent_pipeline/agent_pipeline.py`](../../src/agentdojo/agent_pipeline/agent_pipeline.py) (`dfc_agent_framework_integration` defense)
- Task cleanup: [`src/agentdojo/task_suite/task_suite.py`](../../src/agentdojo/task_suite/task_suite.py)
- Tests: [`dfc_agent_framework_integration/tests/`](../../dfc_agent_framework_integration/tests/), [`tests/test_agent_pipeline/test_dfc_agent_framework_integration.py`](../../tests/test_agent_pipeline/test_dfc_agent_framework_integration.py)
