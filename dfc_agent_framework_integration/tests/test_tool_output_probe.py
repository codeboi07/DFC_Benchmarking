from __future__ import annotations

import duckdb

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from dfc_agent_framework_integration.events import record_tool_output_row
from dfc_agent_framework_integration.schema import RuntimeSchema
from dfc_agent_framework_integration.tool_output_probe import (
    columns_from_flat_dict,
    is_flat_scalar_dict,
    normalize_dict_key_to_column,
)


def test_normalize_dict_key_to_column_handles_spaces():
    assert normalize_dict_key_to_column("First Name") == "first_name"
    assert normalize_dict_key_to_column("Bank Account Number") == "bank_account_number"


def test_is_flat_scalar_dict_rejects_nested_values():
    assert is_flat_scalar_dict({"email": "a@b.com", "password": "secret"})
    assert not is_flat_scalar_dict({"item": {"product_id": "P001"}})
    assert not is_flat_scalar_dict({})


def test_columns_from_flat_dict_builds_source_key_map():
    columns, descriptions, source_keys = columns_from_flat_dict({"First Name": "Alice", "Email": "alice@example.com"})
    assert columns["first_name"] == "VARCHAR"
    assert columns["email"] == "VARCHAR"
    assert source_keys["first_name"] == "First Name"
    assert "first_name" in descriptions


def test_shopping_user_information_output_is_flattened_from_probe():
    suite = get_suites("v1.2.2")["shopping"]
    env = suite.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(suite.tools)
    schema = RuntimeSchema.from_tools(runtime.functions, functions_runtime=runtime, env=env)
    output = next(
        relation
        for relation in schema.tool_output_relations
        if relation.tool_name == "get_shopping_account_user_information"
    )
    assert "__dfc_raw_json" not in output.columns
    assert "password" in output.columns
    assert "bank_account_number" in output.columns
    assert output.source_key_by_column["password"] == "Password"
    assert output.description
    assert "user information" in output.description.lower()
    input_relation = next(
        relation
        for relation in schema.tool_input_relations
        if relation.tool_name == "get_shopping_account_user_information"
    )
    assert input_relation.description
    assert "user information" in input_relation.description.lower()


def test_nested_dict_outputs_remain_raw_json():
    suite = get_suites("v1.2.2")["shopping"]
    env = suite.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(suite.tools)
    schema = RuntimeSchema.from_tools(runtime.functions, functions_runtime=runtime, env=env)
    cart_output = next(relation for relation in schema.tool_output_relations if relation.tool_name == "view_cart")
    assert cart_output.columns == {"__dfc_raw_json": "VARCHAR"}


def test_record_tool_output_row_uses_source_key_mapping():
    sample = {"First Name": "Alice", "Password": "secret"}
    columns, descriptions, source_keys = columns_from_flat_dict(sample)
    raw_conn = duckdb.connect()
    relation_name = "ProbeOutput"
    raw_conn.execute(
        'CREATE TABLE "ProbeOutput" (first_name VARCHAR, password VARCHAR, '
        "__dfc_event_id VARCHAR, __dfc_task_id VARCHAR, __dfc_event_type VARCHAR, "
        "__dfc_status VARCHAR, __dfc_created_at VARCHAR, __dfc_error VARCHAR)"
    )
    record_tool_output_row(
        raw_conn,
        relation_name,
        columns,
        sample,
        event_id="event-1",
        task_id="task-1",
        tool_name="probe_tool",
        source_key_by_column=source_keys,
    )
    row = raw_conn.execute('SELECT first_name, password FROM "ProbeOutput"').fetchone()
    assert row == ("Alice", "secret")
