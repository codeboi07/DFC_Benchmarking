from __future__ import annotations

import re
from typing import Any

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import (
    policy_generation_instructions,
    sink_classification_instructions,
)
from dfc_agent_framework_integration.repair import register_policy_with_repair
from dfc_agent_framework_integration.schema import (
    DeletedPolicyRecord,
    GeneratedPolicy,
    GeneratedPolicySet,
    PolicyRegistrationRecord,
    RuntimeSchema,
    SinkEffectClassification,
)

_SINK_RE = re.compile(r"^\s*SINK\s+(\w+)", re.IGNORECASE | re.MULTILINE)
_PREAMBLE_REF_RE = re.compile(r"PreambleData\.(\w+)")
# Structure that marks a value as a concrete, verbatim-matchable target (so an equality can hold):
# contains @, a slash/colon/backslash, a digit, or a dotted file-extension-like suffix.
_STRUCTURED_VALUE_RE = re.compile(r"[@/:\\]|\d|\.[A-Za-z]{1,5}\b")


def _is_precise_value(value: str) -> bool:
    """A precise value is one a sink field could equal verbatim: a single token (no whitespace), a
    structured identifier (email/url/path/number/filename), or a proper-noun/title-cased name. A
    lowercase multi-word natural-language phrase ("smart watch", "buy groceries") is a goal *description*,
    not a target — grounding an equality on it only causes false positives. Conservative on purpose:
    when in doubt it returns True (precise) so we never drop a real guard."""
    v = value.strip()
    if not v or " " not in v:
        return True
    if _STRUCTURED_VALUE_RE.search(v):
        return True
    words = [w for w in v.split() if any(ch.isalpha() for ch in w)]
    return bool(words) and all(w[0].isupper() for w in words)


def _policy_sink_relation(generated: GeneratedPolicy) -> str | None:
    """The sink relation this policy targets — prefer the declared field, fall back to parsing the PGN."""
    if generated.applies_to_relation:
        return generated.applies_to_relation
    match = _SINK_RE.search(generated.pgn or "")
    return match.group(1) if match else None


def _classify_read_only_sinks(
    llm: StructuredLLMClient,
    *,
    model: str,
    runtime_schema: RuntimeSchema,
    event_log: DFCEventLog,
) -> set[str]:
    """Layer 2: neutral classifier marks each sink read_only vs effectful. Returns the set of relations
    judged read_only. Fail-safe: on any error, return an empty set (treat everything as effectful → keep
    all policies), because dropping an effectful policy would silently weaken security."""
    if not runtime_schema.tool_input_relations:
        return set()
    try:
        result = llm.parse(
            model=model,
            instructions=sink_classification_instructions(runtime_schema),
            input_text="Classify each sink relation as read_only or effectful.",
            text_format=SinkEffectClassification,
        )
    except Exception as exc:  # noqa: BLE001 — fail-safe: never let classification crash policy setup
        event_log.log("sink_classification_failed", error=f"{type(exc).__name__}: {exc}")
        return set()
    read_only = {s.relation for s in result.sinks if s.kind == "read_only"}
    event_log.log(
        "sink_classification_complete",
        read_only=sorted(read_only),
        effectful=sorted(s.relation for s in result.sinks if s.kind == "effectful"),
    )
    return read_only


def _goal_phrase_groundings(generated: GeneratedPolicy, preamble_facts: dict[str, str]) -> list[str]:
    """Layer 3: list the preamble facts a policy grounds on whose value is a goal phrase (loose
    description) rather than a precise target. A non-empty result means the equality cannot hold across
    the agent's legitimate values, so the policy will only false-positive."""
    flagged: list[str] = []
    for fact_key in set(_PREAMBLE_REF_RE.findall(generated.pgn or "")):
        value = preamble_facts.get(fact_key)
        if value is None:
            continue
        if not _is_precise_value(value):
            flagged.append(f"{fact_key}={value!r}")
    return flagged


def generate_and_register_policies(
    conn: Any,
    llm: StructuredLLMClient,
    *,
    model: str,
    preamble: str,
    preamble_facts: dict[str, str],
    runtime_schema: RuntimeSchema,
    event_log: DFCEventLog | None = None,
    classifier_model: str | None = None,
) -> tuple[list[GeneratedPolicy], list[str], list[DeletedPolicyRecord], list[PolicyRegistrationRecord]]:
    event_log = event_log or DFCEventLog(None)
    generated_set = llm.parse(
        model=model,
        instructions=policy_generation_instructions(runtime_schema, preamble_facts),
        input_text=preamble,
        text_format=GeneratedPolicySet,
    )
    event_log.log("policy_generation_llm_complete", policy_count=len(generated_set.policies))

    # Layer 2 — deterministic admission control: drop policies on read-only sinks (no security value,
    # they only break legitimate reads/searches). The classifier is a separate, neutral call.
    read_only_sinks = _classify_read_only_sinks(
        llm,
        model=classifier_model or model,
        runtime_schema=runtime_schema,
        event_log=event_log,
    )
    admitted: list[GeneratedPolicy] = []
    for generated in generated_set.policies:
        sink = _policy_sink_relation(generated)
        if sink is not None and sink in read_only_sinks:
            event_log.log(
                "policy_dropped_read_only_sink",
                policy_id=generated.policy_id,
                sink=sink,
            )
            continue
        # Layer 3 — drop policies that ground an equality on a goal-phrase fact (a loose description, not
        # a precise target). Such a constraint cannot hold across the agent's legitimate values, so it
        # only produces false positives. The real egress/transfer guards ground on precise values (emails,
        # accounts, urls) or tool-output relations, so they are unaffected.
        goal_phrases = _goal_phrase_groundings(generated, preamble_facts)
        if goal_phrases:
            event_log.log(
                "policy_dropped_goal_phrase_grounding",
                policy_id=generated.policy_id,
                sink=sink,
                goal_phrase_facts=goal_phrases,
            )
            continue
        admitted.append(generated)

    registered_ids: list[str] = []
    deleted: list[DeletedPolicyRecord] = []
    registration_records: list[PolicyRegistrationRecord] = []

    for generated in admitted:
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
    return admitted, registered_ids, deleted, registration_records
