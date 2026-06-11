from __future__ import annotations

from typing import Any

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import policy_generation_instructions
from dfc_agent_framework_integration.repair import register_policy_with_repair
from dfc_agent_framework_integration.schema import (
    DeletedPolicyRecord,
    GeneratedPolicy,
    GeneratedPolicySet,
    PolicyRegistrationRecord,
    RuntimeSchema,
)


def generate_and_register_policies(
    conn: Any,
    llm: StructuredLLMClient,
    *,
    model: str,
    preamble: str,
    preamble_facts: dict[str, str],
    runtime_schema: RuntimeSchema,
    event_log: DFCEventLog | None = None,
) -> tuple[list[GeneratedPolicy], list[str], list[DeletedPolicyRecord], list[PolicyRegistrationRecord]]:
    event_log = event_log or DFCEventLog(None)
    generated_set = llm.parse(
        model=model,
        instructions=policy_generation_instructions(runtime_schema, preamble_facts),
        input_text=preamble,
        text_format=GeneratedPolicySet,
    )
    event_log.log("policy_generation_llm_complete", policy_count=len(generated_set.policies))

    registered_ids: list[str] = []
    deleted: list[DeletedPolicyRecord] = []
    registration_records: list[PolicyRegistrationRecord] = []

    for generated in generated_set.policies:
        event_log.log("policy_register_start", policy_id=generated.policy_id)
        result = register_policy_with_repair(
            conn,
            llm,
            model=model,
            generated=generated,
            preamble=preamble,
            preamble_facts=preamble_facts,
            runtime_schema=runtime_schema,
            event_log=event_log,
        )
        if result.registered is not None:
            registered_ids.append(generated.policy_id)
            event_log.log(
                "policy_register_success",
                policy_id=generated.policy_id,
                repair_attempts=result.repair_attempts,
            )
            registration_records.append(
                PolicyRegistrationRecord(
                    policy_id=generated.policy_id,
                    repair_attempts=result.repair_attempts,
                    outcome="registered",
                )
            )
        elif result.deleted:
            event_log.log(
                "policy_register_deleted",
                policy_id=generated.policy_id,
                repair_attempts=result.repair_attempts,
                rationale=result.delete_rationale,
                error=result.error,
            )
            deleted.append(
                DeletedPolicyRecord(
                    policy_id=generated.policy_id,
                    rationale=result.delete_rationale or "deleted",
                    error=result.error,
                )
            )
            registration_records.append(
                PolicyRegistrationRecord(
                    policy_id=generated.policy_id,
                    repair_attempts=result.repair_attempts,
                    outcome="deleted",
                )
            )
    return generated_set.policies, registered_ids, deleted, registration_records
