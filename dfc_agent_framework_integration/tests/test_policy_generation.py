from __future__ import annotations

from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.schema import BenchmarkTaskContext, RuntimeSchema


def test_policy_generation_registers_valid_policies(send_email_runtime, combined_llm):
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        dfc_model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        assert context.registered_policy_ids == ["send_email_recipient"]
        assert len(context.conn.policies()) == 1
    finally:
        context.close()
