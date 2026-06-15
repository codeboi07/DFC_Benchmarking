from __future__ import annotations

from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.dfc_event_log import DFCEventLog, EVENTS_FILENAME
from dfc_agent_framework_integration.schema import BenchmarkTaskContext, RuntimeSchema


def test_prepare_task_writes_incremental_events(tmp_path, combined_llm, send_email_runtime):
    event_log = DFCEventLog(tmp_path)
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            task_id="user_task_0",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        dfc_model="fake-model",
        functions=send_email_runtime.functions,
        event_log=event_log,
    )
    try:
        events_path = tmp_path / EVENTS_FILENAME
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        event_names = [line.split('"event": "')[1].split('"')[0] for line in lines if '"event": "' in line]
        assert "prepare_task_start" in event_names
        assert "extraction_complete" in event_names
        assert "policy_generation_complete" in event_names
        assert "prepare_task_complete" in event_names
        assert context.event_log is event_log
    finally:
        context.close()
