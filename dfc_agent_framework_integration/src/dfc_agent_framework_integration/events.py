from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from dfc_agent_framework_integration.schema import RelationSchema
from dfc_agent_framework_integration.tools import BenchmarkTool

DFC_PREFIX = "__dfc_"
PREAMBLE_RELATION = "PreambleData"
PROMPT_INPUT_RELATION = "PromptInput"
ASSISTANT_RESPONSE_RELATION = "AssistantResponseOutput"

METADATA_COLUMNS: dict[str, str] = {
    "__dfc_event_id": "VARCHAR",
    "__dfc_task_id": "VARCHAR",
    "__dfc_event_type": "VARCHAR",
    "__dfc_status": "VARCHAR",
    "__dfc_created_at": "VARCHAR",
    "__dfc_error": "VARCHAR",
}

LIST_FIELD_SINGULAR_COLUMNS: dict[str, str] = {
    "recipients": "recipient",
    "cc": "cc",
    "bcc": "bcc",
}

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def tool_name_to_table_base(tool_name: str) -> str:
    if not tool_name.strip():
        raise ValueError("Tool name must not be empty")
    if DFC_PREFIX in tool_name:
        raise ValueError(f"Tool name must not contain reserved prefix {DFC_PREFIX!r}")
    parts = re.split(r"[_\-\s]+", tool_name.strip())
    pascal = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not pascal or not _SQL_IDENTIFIER.match(pascal):
        raise ValueError(f"Tool name {tool_name!r} does not produce a valid table base name")
    return pascal


def input_table_name(tool_name: str) -> str:
    return f"{tool_name_to_table_base(tool_name)}Input"


def output_table_name(tool_name: str) -> str:
    return f"{tool_name_to_table_base(tool_name)}Output"


def quote_identifier(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def _is_string_array(property_schema: dict[str, Any]) -> bool:
    if property_schema.get("type") == "array":
        items = property_schema.get("items", {})
        return isinstance(items, dict) and items.get("type") == "string"
    return False


def list_singular_column(field_name: str) -> str | None:
    if field_name in LIST_FIELD_SINGULAR_COLUMNS:
        return LIST_FIELD_SINGULAR_COLUMNS[field_name]
    if field_name.endswith("s") and len(field_name) > 1:
        return field_name[:-1]
    return None


def _description_from_property(property_schema: dict[str, Any]) -> str | None:
    description = property_schema.get("description")
    if isinstance(description, str):
        stripped = description.strip()
        if stripped:
            return stripped
    return None


def column_specs_from_function(function: BenchmarkTool) -> tuple[dict[str, str], dict[str, str]]:
    schema = function.parameters.model_json_schema()
    properties = schema.get("properties", {})
    columns: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for name, property_schema in properties.items():
        if name.startswith(DFC_PREFIX) or name.startswith("__passant_"):
            raise ValueError(f"Field name {name!r} collides with reserved prefix")
        prop = property_schema if isinstance(property_schema, dict) else {}
        columns[name] = "VARCHAR"
        field_description = _description_from_property(prop)
        if field_description is not None:
            descriptions[name] = field_description
        if _is_string_array(prop):
            singular = list_singular_column(name)
            if singular is not None:
                columns[singular] = "VARCHAR"
                if field_description is not None:
                    descriptions[singular] = (
                        f"{field_description} "
                        f"(singular column materialized from list parameter {name!r}; one staged row per element)"
                    )
    return columns, descriptions


def column_specs_from_return_type(function: BenchmarkTool) -> tuple[dict[str, str], dict[str, str]]:
    return_type = function.return_type
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        schema = return_type.model_json_schema()
        properties = schema.get("properties", {})
        columns: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        for name, property_schema in properties.items():
            if name.startswith(DFC_PREFIX) or name.startswith("__passant_"):
                continue
            prop = property_schema if isinstance(property_schema, dict) else {}
            columns[name] = "VARCHAR"
            field_description = _description_from_property(prop)
            if field_description is not None:
                descriptions[name] = field_description
        if columns:
            return columns, descriptions

    return (
        {"__dfc_raw_json": "VARCHAR"},
        {
            "__dfc_raw_json": (
                f"Serialized tool return value for {function.name!r} "
                f"({function.description.splitlines()[0]})"
            )
        },
    )


def fields_from_function(function: BenchmarkTool) -> dict[str, str]:
    columns, _ = column_specs_from_function(function)
    return columns


def list_fields_for_relation(relation_columns: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for field_name in relation_columns:
        singular = list_singular_column(field_name)
        if singular is not None and singular in relation_columns and singular != field_name:
            mapping[field_name] = singular
    return mapping


def prompt_input_relation() -> RelationSchema:
    return RelationSchema(
        name=PROMPT_INPUT_RELATION,
        columns={"content": "VARCHAR", "target": "VARCHAR"},
        column_descriptions={
            "content": "Prompt or user message text sent to a model or sub-agent.",
            "target": "Optional target name for the prompt recipient model or agent.",
        },
    )


def assistant_response_relation() -> RelationSchema:
    return RelationSchema(
        name=ASSISTANT_RESPONSE_RELATION,
        columns={"content": "VARCHAR"},
        column_descriptions={
            "content": "Final assistant response text validated before task completion.",
        },
    )


def normalize_tool_names(tool_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    normalized_to_tool: dict[str, str] = {}
    for name in tool_names:
        base = tool_name_to_table_base(name)
        key = base.lower()
        if key in normalized_to_tool and normalized_to_tool[key] != name:
            raise ValueError(
                f"Tool names {normalized_to_tool[key]!r} and {name!r} collide after normalization"
            )
        normalized_to_tool[key] = name
        mapping[name] = base
    return mapping


def staging_table_name(relation_name: str) -> str:
    return f"{relation_name}WriteStaging"


def _create_staging_table(raw_conn: Any, relation: RelationSchema) -> None:
    staging_name = staging_table_name(relation.name)
    raw_conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(staging_name)}")
    columns = dict(relation.columns)
    for meta_name, meta_type in METADATA_COLUMNS.items():
        columns[meta_name] = meta_type
    if not columns:
        raw_conn.execute(f"CREATE TABLE {quote_identifier(staging_name)} ()")
        return
    col_defs = ", ".join(
        f"{quote_identifier(name)} {col_type}" for name, col_type in columns.items()
    )
    raw_conn.execute(f"CREATE TABLE {quote_identifier(staging_name)} ({col_defs})")


def ensure_relation_write_tables(
    raw_conn: Any,
    relation_name: str,
    relation_columns: dict[str, str],
) -> None:
    relation = RelationSchema(name=relation_name, columns=relation_columns)
    _create_relation_table(raw_conn, relation)
    _create_staging_table(raw_conn, relation)


def build_probe_payload(
    relation_columns: dict[str, str],
    preamble_facts: dict[str, str],
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for column in relation_columns:
        if column.startswith("__dfc_"):
            continue
        value: str | None = None
        for key, fact in preamble_facts.items():
            if column in key or key.endswith(column):
                value = fact
                break
        if value is None:
            value = preamble_facts.get(f"authorized_{column}") or preamble_facts.get(column)
        if value is None and preamble_facts:
            value = next(iter(preamble_facts.values()))
        payload[column] = value or f"__dfc_probe_{column}"
    return payload


def write_event_row(
    dfc_conn: Any,
    raw_conn: Any,
    relation_name: str,
    row: dict[str, Any],
    relation_columns: dict[str, str],
) -> None:
    staging = staging_table_name(relation_name)
    clear_relation_rows(raw_conn, staging)
    insert_event_row(raw_conn, staging, row)

    insert_columns = list(relation_columns.keys()) + [
        column for column in row if column.startswith("__dfc_")
    ]
    columns_sql = ", ".join(quote_identifier(column) for column in insert_columns)
    select_sql = ", ".join(f"s.{quote_identifier(column)}" for column in insert_columns)
    sql = (
        f"INSERT INTO {quote_identifier(relation_name)} ({columns_sql}) "
        f"SELECT {select_sql} FROM {quote_identifier(staging)} AS s "
        f"CROSS JOIN {quote_identifier(PREAMBLE_RELATION)}"
    )
    dfc_conn.execute(sql)


def attempt_write_event_row(
    dfc_conn: Any,
    raw_conn: Any,
    relation_name: str,
    row: dict[str, Any],
    relation_columns: dict[str, str],
) -> tuple[bool, str | None]:
    try:
        write_event_row(dfc_conn, raw_conn, relation_name, row, relation_columns)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def create_event_tables(raw_conn: Any, runtime_schema: Any) -> None:
    for relation in runtime_schema.all_relations():
        _create_relation_table(raw_conn, relation)
        _create_staging_table(raw_conn, relation)
    _create_relation_table(
        raw_conn,
        RelationSchema(name=PREAMBLE_RELATION, columns={}),
        allow_empty_columns=True,
    )


def _create_relation_table(
    raw_conn: Any,
    relation: RelationSchema,
    *,
    allow_empty_columns: bool = False,
) -> None:
    columns = dict(relation.columns)
    for meta_name, meta_type in METADATA_COLUMNS.items():
        columns[meta_name] = meta_type
    if not columns and not allow_empty_columns:
        raise ValueError(f"Relation {relation.name!r} has no columns")
    if not columns:
        ddl = f"CREATE TABLE {quote_identifier(relation.name)} ()"
        raw_conn.execute(ddl)
        return
    col_defs = ", ".join(
        f"{quote_identifier(name)} {col_type}" for name, col_type in columns.items()
    )
    ddl = f"CREATE TABLE {quote_identifier(relation.name)} ({col_defs})"
    raw_conn.execute(ddl)


def new_event_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_scalar_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return json.dumps(value)
    return str(value)


def expand_payload_rows(
    payload: dict[str, Any],
    relation_columns: dict[str, str],
) -> list[dict[str, Any]]:
    list_fields = list_fields_for_relation(relation_columns)
    explode_field: str | None = None
    explode_values: list[str] = []

    for list_field, singular_field in list_fields.items():
        if list_field not in payload:
            continue
        raw_value = payload[list_field]
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = raw_value
        else:
            parsed = raw_value
        if isinstance(parsed, list):
            values = [str(item) for item in parsed]
        else:
            values = [str(parsed)]
        if values:
            explode_field = list_field
            explode_values = values
            break

    if explode_field is None:
        return [payload]

    list_field, singular_field = explode_field, list_fields[explode_field]
    rows: list[dict[str, Any]] = []
    for value in explode_values:
        row = dict(payload)
        row[singular_field] = value
        row[list_field] = json.dumps(explode_values)
        rows.append(row)
    return rows


def build_insert_row(
    payload: dict[str, Any],
    relation_columns: dict[str, str],
    *,
    event_id: str,
    task_id: str | None,
    event_type: str,
    status: str = "pending",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "__dfc_event_id": event_id,
        "__dfc_task_id": task_id or "",
        "__dfc_event_type": event_type,
        "__dfc_status": status,
        "__dfc_created_at": utc_now_iso(),
        "__dfc_error": None,
    }
    for column in relation_columns:
        if column in payload:
            row[column] = serialize_scalar_value(payload[column])
        else:
            row[column] = None
    return row


def insert_event_row(raw_conn: Any, relation_name: str, row: dict[str, Any]) -> None:
    columns = list(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    quoted_cols = ", ".join(quote_identifier(column) for column in columns)
    values = [row[column] for column in columns]
    sql = f"INSERT INTO {quote_identifier(relation_name)} ({quoted_cols}) VALUES ({placeholders})"
    raw_conn.execute(sql, values)


def clear_relation_rows(raw_conn: Any, relation_name: str) -> None:
    raw_conn.execute(f"DELETE FROM {quote_identifier(relation_name)}")
