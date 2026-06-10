from __future__ import annotations

import re
from typing import Any

import duckdb
from data_flow_control import Policy, dfc

from dfc_agent_framework_integration.events import (
    PREAMBLE_RELATION,
    attempt_write_event_row,
    ensure_relation_write_tables,
)
from dfc_agent_framework_integration.materialize import materialize_preamble_data
from dfc_agent_framework_integration.schema import GeneratedPolicy

_DDL = re.compile(r"^\s*(CREATE|DROP|ALTER)\s+", re.IGNORECASE)
_DESTRUCTIVE = re.compile(r"^\s*(DELETE|UPDATE|TRUNCATE)\s+", re.IGNORECASE)
_METADATA_COLUMNS = {
    "__dfc_event_id",
    "__dfc_task_id",
    "__dfc_event_type",
    "__dfc_status",
    "__dfc_created_at",
    "__dfc_error",
}


def validate_debug_fetch_sql(sql: str) -> str:
    stripped = sql.strip()
    if not stripped:
        raise ValueError("SQL must not be empty")
    if ";" in stripped.rstrip(";"):
        raise ValueError("Only a single SQL statement is allowed")
    if _DDL.match(stripped) or _DESTRUCTIVE.match(stripped):
        raise ValueError("fetchall only supports SELECT statements")
    if not stripped.lower().startswith("select"):
        raise ValueError("fetchall only supports SELECT statements")
    return stripped


def policies_for_relation(
    generated_policies: list[GeneratedPolicy],
    registered_policy_ids: list[str],
    relation_name: str,
) -> list[str]:
    registered = set(registered_policy_ids)
    descriptions: list[str] = []
    for policy in generated_policies:
        if policy.policy_id not in registered:
            continue
        if policy.applies_to_relation == relation_name and policy.description:
            descriptions.append(policy.description)
    return descriptions


def identify_violated_policy_descriptions(
    *,
    relation_name: str,
    attempted_row: dict[str, Any],
    relation_columns: dict[str, str],
    generated_policies: list[GeneratedPolicy],
    registered_policy_ids: list[str],
    preamble_facts: dict[str, str],
    raw_error: str | None = None,
) -> list[str]:
    grouped = policies_for_relation(generated_policies, registered_policy_ids, relation_name)
    if not grouped:
        return ["Data flow policy violation."]

    exact: list[str] = []
    for policy in generated_policies:
        if policy.policy_id not in registered_policy_ids:
            continue
        if policy.applies_to_relation != relation_name:
            continue
        try:
            scratch_raw = duckdb.connect()
            materialize_preamble_data(scratch_raw, preamble_facts, PREAMBLE_RELATION)
            ensure_relation_write_tables(scratch_raw, relation_name, relation_columns)
            scratch = dfc(scratch_raw)
            scratch.register_policy(Policy.from_pgn(policy.pgn))
            allowed, _ = attempt_write_event_row(
                scratch,
                scratch_raw,
                relation_name,
                attempted_row,
                relation_columns,
            )
            if not allowed and policy.description:
                exact.append(policy.description)
            scratch.close()
        except Exception:
            continue

    if exact:
        return exact
    if grouped:
        return grouped
    if raw_error:
        return [f"Data flow policy violation: {raw_error}"]
    return ["Data flow policy violation."]
