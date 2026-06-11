from __future__ import annotations

from typing import Any

from data_flow_control import Policy

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.llm import StructuredLLMClient
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
        repair_attempts: int = 0,
    ) -> None:
        self.registered = registered
        self.deleted = deleted
        self.delete_rationale = delete_rationale
        self.error = error
        self.repair_attempts = repair_attempts


def known_relation_names(runtime_schema: RuntimeSchema) -> set[str]:
    return runtime_schema.relation_names()


def validate_policy_catalog(policy: Policy, runtime_schema: RuntimeSchema) -> None:
    known = known_relation_names(runtime_schema)
    if policy.sink and policy.sink not in known:
        raise ValueError(f"Sink relation {policy.sink!r} does not exist in runtime schema")
    for relation_name in policy.dimensions:
        if relation_name not in known:
            raise ValueError(f"Dimension table {relation_name!r} does not exist in runtime schema")
    for relation_name in policy.sources + policy.required_sources:
        if relation_name not in known:
            raise ValueError(f"Source relation {relation_name!r} does not exist in runtime schema")


def register_policy_with_repair(
    conn: Any,
    llm: StructuredLLMClient,
    *,
    model: str,
    generated: GeneratedPolicy,
    preamble: str,
    preamble_facts: dict[str, str],
    runtime_schema: RuntimeSchema,
    max_repair_attempts: int = 2,
    event_log: DFCEventLog | None = None,
) -> PolicyRegistrationResult:
    event_log = event_log or DFCEventLog(None)
    pgn = generated.pgn
    description = generated.description
    last_error: str | None = None
    repair_attempts = 0

    for attempt in range(max_repair_attempts + 1):
        try:
            parsed = Policy.from_pgn(pgn)
            validate_policy_catalog(parsed, runtime_schema)
            conn.register_policy(parsed)
            if description and not parsed.description:
                parsed.description = description
            event_log.log(
                "policy_register_attempt_success",
                policy_id=generated.policy_id,
                attempt=attempt,
                repair_attempts=repair_attempts,
            )
            return PolicyRegistrationResult(registered=parsed, repair_attempts=repair_attempts)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            event_log.log(
                "policy_register_attempt_failed",
                policy_id=generated.policy_id,
                attempt=attempt,
                error=last_error,
            )
            if attempt >= max_repair_attempts:
                break
            event_log.log(
                "policy_repair_start",
                policy_id=generated.policy_id,
                attempt=attempt + 1,
                error=last_error,
            )
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
            repair_attempts += 1
            if decision.delete:
                event_log.log(
                    "policy_repair_deleted",
                    policy_id=generated.policy_id,
                    rationale=decision.rationale,
                )
                return PolicyRegistrationResult(
                    deleted=True,
                    delete_rationale=decision.rationale,
                    error=last_error,
                    repair_attempts=repair_attempts,
                )
            if not decision.repaired_pgn:
                event_log.log(
                    "policy_repair_missing_pgn",
                    policy_id=generated.policy_id,
                    rationale=decision.rationale,
                )
                return PolicyRegistrationResult(
                    deleted=True,
                    delete_rationale=decision.rationale or "Repair response missing repaired_pgn",
                    error=last_error,
                    repair_attempts=repair_attempts,
                )
            pgn = decision.repaired_pgn
            if decision.repaired_description:
                description = decision.repaired_description
            event_log.log(
                "policy_repair_complete",
                policy_id=generated.policy_id,
                attempt=repair_attempts,
            )

    return PolicyRegistrationResult(
        deleted=True,
        delete_rationale="Exceeded repair attempts",
        error=last_error,
        repair_attempts=repair_attempts,
    )
