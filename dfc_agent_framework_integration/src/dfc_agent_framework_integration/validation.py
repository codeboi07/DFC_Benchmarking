from __future__ import annotations

import re

from dfc_agent_framework_integration.schema import GeneratedPolicy, ViolationIdentification

_DDL = re.compile(r"^\s*(CREATE|DROP|ALTER)\s+", re.IGNORECASE)
_DESTRUCTIVE = re.compile(r"^\s*(DELETE|UPDATE|TRUNCATE)\s+", re.IGNORECASE)


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
) -> tuple[list[str], list[str]]:
    registered = set(registered_policy_ids)
    policy_ids: list[str] = []
    descriptions: list[str] = []
    for policy in generated_policies:
        if policy.policy_id not in registered:
            continue
        if policy.applies_to_relation == relation_name:
            policy_ids.append(policy.policy_id)
            if policy.description:
                descriptions.append(policy.description)
    return policy_ids, descriptions


def identify_violated_policies(
    *,
    relation_name: str,
    generated_policies: list[GeneratedPolicy],
    registered_policy_ids: list[str],
    raw_error: str | None = None,
) -> ViolationIdentification:
    policy_ids, descriptions = policies_for_relation(
        generated_policies,
        registered_policy_ids,
        relation_name,
    )
    if policy_ids:
        return ViolationIdentification(
            policy_ids=policy_ids,
            policy_descriptions=descriptions or ["Data flow policy violation."],
        )
    if raw_error:
        return ViolationIdentification(
            policy_descriptions=[f"Data flow policy violation: {raw_error}"],
        )
    return ViolationIdentification(
        policy_descriptions=["Data flow policy violation."],
    )


def identify_violated_policy_descriptions(
    *,
    relation_name: str,
    generated_policies: list[GeneratedPolicy],
    registered_policy_ids: list[str],
    raw_error: str | None = None,
) -> list[str]:
    return identify_violated_policies(
        relation_name=relation_name,
        generated_policies=generated_policies,
        registered_policy_ids=registered_policy_ids,
        raw_error=raw_error,
    ).policy_descriptions
