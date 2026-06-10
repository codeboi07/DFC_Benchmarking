from __future__ import annotations

from typing import Any

from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import policy_generation_instructions
from dfc_agent_framework_integration.repair import register_policy_with_repair
from dfc_agent_framework_integration.schema import (
    DeletedPolicyRecord,
    GeneratedPolicy,
    GeneratedPolicySet,
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
    raw_conn: Any | None = None,
) -> tuple[list[GeneratedPolicy], list[str], list[DeletedPolicyRecord]]:
    generated_set = llm.parse(
        model=model,
        instructions=policy_generation_instructions(runtime_schema, preamble_facts),
        input_text=preamble,
        text_format=GeneratedPolicySet,
    )
    registered_ids: list[str] = []
    deleted: list[DeletedPolicyRecord] = []

    for generated in generated_set.policies:
        result = register_policy_with_repair(
            conn,
            llm,
            model=model,
            generated=generated,
            preamble=preamble,
            preamble_facts=preamble_facts,
            runtime_schema=runtime_schema,
            raw_conn=raw_conn,
        )
        if result.registered is not None:
            registered_ids.append(generated.policy_id)
        elif result.deleted:
            deleted.append(
                DeletedPolicyRecord(
                    policy_id=generated.policy_id,
                    rationale=result.delete_rationale or "deleted",
                    error=result.error,
                )
            )
    return generated_set.policies, registered_ids, deleted
