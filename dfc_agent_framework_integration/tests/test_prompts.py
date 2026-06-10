from __future__ import annotations

from dfc_agent_framework_integration.prompts import format_schema_for_policy_generation, policy_generation_instructions
from dfc_agent_framework_integration.schema import RuntimeSchema


def test_format_schema_includes_tool_param_descriptions(send_email_runtime):
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    schema_text = format_schema_for_policy_generation(
        runtime_schema,
        {"authorized_recipient_email": "alice@example.com"},
    )

    assert "Tool input relations:" in schema_text
    assert "Tool output relations:" in schema_text
    assert "Relation SendEmailInput:" in schema_text
    assert "Relation SendEmailOutput:" in schema_text
    assert "recipients: VARCHAR" in schema_text
    assert "recipient: VARCHAR" in schema_text
    assert "email addresses" in schema_text.lower() or "recipient" in schema_text.lower()
    assert "authorized_recipient_email: 'alice@example.com'" in schema_text


def test_policy_generation_instructions_require_on_fail_kill(send_email_runtime):
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    instructions = policy_generation_instructions(
        runtime_schema,
        {"authorized_recipient_email": "alice@example.com"},
    )

    assert "Always use ON FAIL KILL" in instructions
    assert "Do not use ON FAIL REMOVE" in instructions
    assert "SINK <ToolInput> DIMENSION PreambleData" in instructions


def test_runtime_schema_carries_column_descriptions(send_email_runtime):
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    send_email_input = runtime_schema.tool_input_relations[0]
    assert send_email_input.name == "SendEmailInput"
    assert "recipients" in send_email_input.column_descriptions
    assert "recipient" in send_email_input.column_descriptions

    send_email_output = runtime_schema.tool_output_relations[0]
    assert send_email_output.name == "SendEmailOutput"
    assert "__dfc_raw_json" in send_email_output.columns
    assert "__dfc_raw_json" in send_email_output.column_descriptions
