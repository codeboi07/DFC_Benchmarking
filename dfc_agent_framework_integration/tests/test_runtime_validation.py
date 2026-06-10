from __future__ import annotations

import pytest

from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.events import fields_from_function
from dfc_agent_framework_integration.schema import BenchmarkTaskContext, RuntimeSchema


def test_sql_validation_rejects_unsafe_statements():
    from dfc_agent_framework_integration.validation import validate_debug_fetch_sql

    with pytest.raises(ValueError):
        validate_debug_fetch_sql("DELETE FROM SendEmailInput")


def test_send_email_schema_exposes_recipient_column(send_email_function):
    columns = fields_from_function(send_email_function)
    assert "recipients" in columns
    assert "recipient" in columns


def test_blocked_tool_call_returns_policy_description(send_email_runtime, combined_llm):
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        violation = context.validate_tool_call(
            "send_email",
            {"recipients": ["trustme@gmail.com"], "subject": "hi", "body": "hello"},
        )
        assert violation is not None
        assert any("authorized" in desc.lower() for desc in violation.policy_descriptions)
    finally:
        context.close()


def test_allowed_tool_call_with_list_recipients(send_email_runtime, combined_llm):
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        violation = context.validate_tool_call(
            "send_email",
            {"recipients": ["alice@example.com"], "subject": "hi", "body": "hello"},
        )
        assert violation is None
    finally:
        context.close()


def test_multiple_recipients_block_if_any_unauthorized(send_email_runtime, combined_llm):
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        violation = context.validate_tool_call(
            "send_email",
            {
                "recipients": ["alice@example.com", "trustme@gmail.com"],
                "subject": "hi",
                "body": "hello",
            },
        )
        assert violation is not None
    finally:
        context.close()
