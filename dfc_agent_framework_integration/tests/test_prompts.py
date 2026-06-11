from __future__ import annotations

from dfc_agent_framework_integration.prompts import (
    format_schema_for_policy_generation,
    policy_generation_instructions,
    repair_instructions,
)
from dfc_agent_framework_integration.schema import GeneratedPolicy, PolicyRepairDecision, RuntimeSchema


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

    assert "Structured output format" in instructions
    assert "Put the full executable policy in each object's `pgn` field" in instructions
    assert "Policies whose `pgn` omits CONSTRAINT" in instructions
    pgn_field = GeneratedPolicy.model_fields["pgn"]
    assert pgn_field.description is not None
    assert "Do not put only the SINK line" in pgn_field.description
    assert "Always use ON FAIL KILL" in instructions
    assert "Do not use ON FAIL REMOVE" in instructions
    assert "DIMENSION may reference any relation in the schema" in instructions
    assert "Do not wrap SINK columns in aggregates" in instructions
    assert "SOURCE columns may need aggregates" in instructions
    assert "Data exfiltration policies" in instructions
    assert "Tool: send_email" in instructions
    assert "Description:" in instructions
    assert "Prevent data exfiltration" in instructions
    assert "GetShoppingAccountUserInformationOutput" in instructions
    assert "COALESCE(max(password), '')" in instructions
    assert "DIMENSION (SELECT COALESCE" in instructions
    assert "AccountInfo.password" in instructions
    assert "contains(SendEmailInput.body, AccountInfo.password)" in instructions
    assert "InputToWebpageInput" in instructions


def test_repair_instructions_include_policy_examples(send_email_runtime):
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    instructions = repair_instructions(
        runtime_schema,
        preamble="Email alice@example.com",
        preamble_facts={"authorized_recipient_email": "alice@example.com"},
        failed_pgn="SINK SendEmailInput",
        error_class="ValueError",
        error_message="missing required clause: CONSTRAINT",
    )

    assert "Example 1" in instructions
    assert "Example 5" in instructions
    assert "Example 10" not in instructions
    assert "Failed PGN:\nSINK SendEmailInput" in instructions
    assert "Error: ValueError: missing required clause: CONSTRAINT" in instructions
    assert "Use the examples as repair patterns" in instructions
    assert "Repair vs delete" in instructions
    assert "Do NOT delete a policy because the trusted task does not mention" in instructions
    assert "Pre-emptive guards on those sinks are intentional" in instructions
    delete_field = PolicyRepairDecision.model_fields["delete"]
    assert delete_field.description is not None
    assert "pre-emptive external-sink guards should be repaired" in delete_field.description


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
