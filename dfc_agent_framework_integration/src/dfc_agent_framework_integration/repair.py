from __future__ import annotations

from typing import Any

import duckdb
from data_flow_control import Policy, dfc

from dfc_agent_framework_integration.events import (
    PREAMBLE_RELATION,
    attempt_write_event_row,
    build_insert_row,
    build_probe_payload,
    ensure_relation_write_tables,
    new_event_id,
)
from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.materialize import materialize_preamble_data
from dfc_agent_framework_integration.prompts import repair_instructions
from dfc_agent_framework_integration.schema import GeneratedPolicy, PolicyRepairDecision, RuntimeSchema


class PolicyRegistrationResult:
    def __init__(
        self,
        *,
        registered: Policy | None = None,
        deleted: bool = False,
        delete_rationale: str | None = None,
        error: str | None = None,
    ) -> None:
        self.registered = registered
        self.deleted = deleted
        self.delete_rationale = delete_rationale
        self.error = error


def probe_policy_runtime(
    *,
    preamble_facts: dict[str, str],
    pgn: str,
    relation_name: str,
    relation_columns: dict[str, str],
) -> str | None:
    scratch_raw = duckdb.connect()
    try:
        materialize_preamble_data(scratch_raw, preamble_facts, PREAMBLE_RELATION)
        ensure_relation_write_tables(scratch_raw, relation_name, relation_columns)
        scratch = dfc(scratch_raw)
        scratch.register_policy(Policy.from_pgn(pgn))

        probe_payload = build_probe_payload(relation_columns, preamble_facts)
        event_id = new_event_id()
        row = build_insert_row(
            probe_payload,
            relation_columns,
            event_id=event_id,
            task_id=None,
            event_type="policy_probe",
            status="pending",
        )
        allowed, error = attempt_write_event_row(
            scratch,
            scratch_raw,
            relation_name,
            row,
            relation_columns,
        )
        scratch.close()
        if allowed:
            return None
        return error or "Policy probe write was blocked"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        scratch_raw.close()


def register_policy_with_repair(
    conn: Any,
    llm: StructuredLLMClient,
    *,
    model: str,
    generated: GeneratedPolicy,
    preamble: str,
    preamble_facts: dict[str, str],
    runtime_schema: RuntimeSchema,
    raw_conn: Any | None = None,
    max_repair_attempts: int = 2,
) -> PolicyRegistrationResult:
    pgn = generated.pgn
    description = generated.description
    last_error: str | None = None
    relation_columns = {
        relation.name: relation.columns
        for relation in runtime_schema.all_relations()
    }.get(generated.applies_to_relation, {})

    for attempt in range(max_repair_attempts + 1):
        try:
            parsed = Policy.from_pgn(pgn)
            runtime_error = probe_policy_runtime(
                preamble_facts=preamble_facts,
                pgn=pgn,
                relation_name=generated.applies_to_relation,
                relation_columns=relation_columns,
            )
            if runtime_error is not None:
                raise RuntimeError(runtime_error)
            conn.register_policy(parsed)
            if description and not parsed.description:
                parsed.description = description
            return PolicyRegistrationResult(registered=parsed)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_repair_attempts:
                break
            decision = llm.parse(
                model=model,
                instructions=repair_instructions(
                    runtime_schema,
                    preamble,
                    preamble_facts,
                    pgn,
                    type(exc).__name__,
                    str(exc),
                ),
                input_text="Repair or delete the failed policy.",
                text_format=PolicyRepairDecision,
            )
            if decision.delete:
                return PolicyRegistrationResult(
                    deleted=True,
                    delete_rationale=decision.rationale,
                    error=last_error,
                )
            if not decision.repaired_pgn:
                return PolicyRegistrationResult(
                    deleted=True,
                    delete_rationale=decision.rationale or "Repair response missing repaired_pgn",
                    error=last_error,
                )
            pgn = decision.repaired_pgn
            if decision.repaired_description:
                description = decision.repaired_description

    return PolicyRegistrationResult(
        deleted=True,
        delete_rationale="Exceeded repair attempts",
        error=last_error,
    )
