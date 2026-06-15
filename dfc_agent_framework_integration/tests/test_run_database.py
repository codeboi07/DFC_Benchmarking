from __future__ import annotations

import json

import duckdb
from data_flow_control import Policy, dfc
from pydantic import BaseModel, Field

from dfc_agent_framework_integration.events import (
    create_event_tables,
    output_table_name,
    record_tool_output_row,
    serialize_scalar_value,
)
from dfc_agent_framework_integration.materialize import materialize_preamble_data
from dfc_agent_framework_integration.repair import register_policy_with_repair, validate_policy_catalog
from dfc_agent_framework_integration.schema import GeneratedPolicy, RuntimeSchema


def test_validate_policy_catalog_accepts_output_dimension(send_email_runtime):
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    policy = Policy.from_pgn(
        "SINK SendEmailInput DIMENSION SendEmailOutput "
        "CONSTRAINT max(SendEmailInput.recipient) = max(SendEmailOutput.__dfc_raw_json) "
        "ON FAIL KILL DESCRIPTION test"
    )
    validate_policy_catalog(policy, runtime_schema)


def test_register_policy_with_output_dimension(send_email_runtime, combined_llm):
    raw_conn = duckdb.connect()
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    create_event_tables(raw_conn, runtime_schema)
    materialize_preamble_data(raw_conn, {"authorized_recipient_email": "alice@example.com"})
    conn = dfc(raw_conn)

    generated = GeneratedPolicy(
        policy_id="cross_step",
        pgn=(
            "SINK SendEmailInput\n"
            "DIMENSION SendEmailOutput\n"
            "CONSTRAINT max(SendEmailInput.recipient) = max(SendEmailOutput.__dfc_raw_json)\n"
            "ON FAIL KILL\n"
            "DESCRIPTION Cross-step policy"
        ),
        description="Cross-step policy",
        applies_to_relation="SendEmailInput",
        applies_to_event="tool_call:send_email",
        rationale="test",
    )
    result = register_policy_with_repair(
        conn,
        combined_llm,
        model="fake-model",
        generated=generated,
        preamble="Email alice@example.com",
        preamble_facts={"authorized_recipient_email": "alice@example.com"},
        runtime_schema=runtime_schema,
    )
    assert result.registered is not None
    assert result.deleted is False


def test_record_tool_output_populates_output_relation(send_email_runtime):
    raw_conn = duckdb.connect()
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    create_event_tables(raw_conn, runtime_schema)
    relation_name = output_table_name("send_email")
    columns = runtime_schema.tool_output_relations[0].columns
    record_tool_output_row(
        raw_conn,
        relation_name,
        columns,
        "sent:alice@example.com",
        event_id="event-1",
        task_id="task-1",
        tool_name="send_email",
    )
    rows = raw_conn.execute(f'SELECT __dfc_raw_json FROM "{relation_name}"').fetchall()
    assert rows == [("sent:alice@example.com",)]


class _SampleEmail(BaseModel):
    subject: str = Field(description="subject")
    sender: str = Field(description="sender")


def test_serialize_scalar_value_handles_pydantic_model_lists():
    emails = [_SampleEmail(subject="Order confirmed", sender="shop@example.com")]
    serialized = serialize_scalar_value(emails)
    assert serialized is not None
    parsed = json.loads(serialized)
    assert parsed[0]["subject"] == "Order confirmed"
    assert parsed[0]["sender"] == "shop@example.com"


def test_record_tool_output_serializes_list_of_models(send_email_runtime):
    raw_conn = duckdb.connect()
    runtime_schema = RuntimeSchema.from_tools(send_email_runtime.functions)
    create_event_tables(raw_conn, runtime_schema)
    relation_name = output_table_name("send_email")
    columns = runtime_schema.tool_output_relations[0].columns
    emails = [_SampleEmail(subject="Order confirmed", sender="shop@example.com")]
    record_tool_output_row(
        raw_conn,
        relation_name,
        columns,
        emails,
        event_id="event-2",
        task_id="task-1",
        tool_name="send_email",
    )
    rows = raw_conn.execute(f'SELECT __dfc_raw_json FROM "{relation_name}"').fetchall()
    parsed = json.loads(rows[0][0])
    assert parsed[0]["subject"] == "Order confirmed"
