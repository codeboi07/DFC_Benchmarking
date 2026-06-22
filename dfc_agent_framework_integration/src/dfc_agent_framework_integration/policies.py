from __future__ import annotations

import os
import re
from typing import Any

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import (
    policy_generation_instructions,
    revise_policies_instructions,
    sink_classification_instructions,
)
from dfc_agent_framework_integration.repair import register_policy_with_repair, validate_policy_catalog
from dfc_agent_framework_integration.schema import (
    DeletedPolicyRecord,
    GeneratedPolicy,
    GeneratedPolicySet,
    PolicyRegistrationRecord,
    RuntimeSchema,
    SinkEffectClassification,
)

# `<SinkRelation>.<col> = PreambleData.<fact>` — the simple grounding shape we can probe directly.
_GROUNDING_RE = re.compile(r"(\w+)\.(\w+)\s*=\s*PreambleData\.(\w+)")

_SINK_RE = re.compile(r"^\s*SINK\s+(\w+)", re.IGNORECASE | re.MULTILINE)
_PREAMBLE_REF_RE = re.compile(r"PreambleData\.(\w+)")
# Structure that marks a value as a concrete, verbatim-matchable target (so an equality can hold):
# contains @, a slash/colon/backslash, a digit, or a dotted file-extension-like suffix.
_STRUCTURED_VALUE_RE = re.compile(r"[@/:\\]|\d|\.[A-Za-z]{1,5}\b")

# A DEFAULT-DENY (destination grounded on the deny sentinel from context.py) on a URL-visit sink — i.e. the
# guarded column is a url. We scope default-deny to recipient/account sinks (send_money/send_email/...), so these
# URL default-denies are dropped: blanket-blocking every URL breaks legitimate link-following, and off-task URLs
# are handled by named-URL allowlists + the semantic judge, not a blanket deny. A real url ALLOWLIST (grounded on
# a genuine authorized_url fact) does NOT match this and is kept.
_URL_DEFAULT_DENY_RE = re.compile(r"\.\w*url\b\s*=\s*PreambleData\.no_authorized_destination", re.IGNORECASE)


def _is_url_default_deny(policy: GeneratedPolicy) -> bool:
    return bool(_URL_DEFAULT_DENY_RE.search(policy.pgn or ""))


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


def _grounding_targets(
    generated: GeneratedPolicy, sink: str | None, preamble_facts: dict[str, str]
) -> list[tuple[str, str, str]]:
    """`(sink_col, fact_key, legit_value)` for each `sink.col = PreambleData.fact` in this policy where the
    fact actually exists. Only the simple grounding shape is probeable; exfil/`contains()` policies yield []."""
    targets: list[tuple[str, str, str]] = []
    for rel, col, fact in _GROUNDING_RE.findall(generated.pgn or ""):
        if rel == sink and fact in preamble_facts:
            targets.append((col, fact, preamble_facts[fact]))
    return targets


def _probe_one_policy(
    runtime_schema: RuntimeSchema,
    preamble_facts: dict[str, str],
    policy: GeneratedPolicy,
    tool_by_relation: dict[str, str | None],
) -> dict[str, bool]:
    """Register the policy ALONE in a scratch DuckDB and probe it (no LLM): does it ALLOW the task's own
    legitimate value (false_positive if not) and BLOCK a non-matching value (vacuous if not)? Isolating one
    policy per scratch connection keeps attribution clean when several policies share a sink. Only the
    simple grounding shape is probed; everything else returns probed=False."""
    result = {"probed": False, "false_positive": False, "vacuous": False}
    sink = policy.applies_to_relation
    tool = tool_by_relation.get(sink)
    targets = _grounding_targets(policy, sink, preamble_facts)
    if tool is None or not targets:
        return result
    import duckdb
    from data_flow_control import Policy, dfc

    from dfc_agent_framework_integration.events import create_event_tables
    from dfc_agent_framework_integration.materialize import materialize_preamble_data
    from dfc_agent_framework_integration.runtime import DFCRuntimeValidator

    raw = duckdb.connect()
    try:
        create_event_tables(raw, runtime_schema)
        materialize_preamble_data(raw, preamble_facts, runtime_schema.preamble_relation)
        conn = dfc(raw)
        parsed = Policy.from_pgn(policy.pgn)
        validate_policy_catalog(parsed, runtime_schema)
        conn.register_policy(parsed)
        conn.refresh_catalog()
        validator = DFCRuntimeValidator(
            raw, conn, runtime_schema, [policy], [policy.policy_id], preamble_facts
        )
        result["probed"] = True
        for col, _fact, legit in targets:
            if validator.validate_tool_call(tool, {col: legit}) is not None:
                result["false_positive"] = True            # blocks the legitimate value -> bad
            synthetic = f"{legit}_DFCPROBE_X" if legit else "DFCPROBE_X"
            if validator.validate_tool_call(tool, {col: synthetic}) is None:
                result["vacuous"] = True                    # allows an arbitrary value -> useless
    except Exception:
        # parse/registration failures are the repair loop's job, not the probe's
        return {"probed": False, "false_positive": False, "vacuous": False}
    finally:
        raw.close()
    return result


def _admit(
    generated_policies: list[GeneratedPolicy],
    read_only_sinks: set[str],
    preamble_facts: dict[str, str],
    event_log: DFCEventLog,
) -> tuple[list[GeneratedPolicy], list[dict]]:
    """Layer 2 + Layer 3 admission control. Returns (admitted, problems) — problems describe the dropped
    policies so they can be fed back to the generator."""
    admitted: list[GeneratedPolicy] = []
    problems: list[dict] = []
    for generated in generated_policies:
        sink = _policy_sink_relation(generated)
        if sink is not None and sink in read_only_sinks:
            event_log.log("policy_dropped_read_only_sink", policy_id=generated.policy_id, sink=sink)
            problems.append({"policy_id": generated.policy_id, "sink": sink, "reason": "read_only_sink",
                             "detail": f"sink {sink} only reads/searches data; do not constrain it"})
            continue
        goal_phrases = _goal_phrase_groundings(generated, preamble_facts)
        if goal_phrases:
            event_log.log("policy_dropped_goal_phrase_grounding", policy_id=generated.policy_id, sink=sink,
                          goal_phrase_facts=goal_phrases)
            problems.append({"policy_id": generated.policy_id, "sink": sink, "reason": "goal_phrase",
                             "detail": f"grounds on goal-phrase fact(s) {goal_phrases}; pin to a precise value or remove"})
            continue
        admitted.append(generated)
    return admitted, problems


def _drop_uncompilable_policies(
    policies: list[GeneratedPolicy],
    runtime_schema: RuntimeSchema,
    preamble_facts: dict[str, str],
    event_log: DFCEventLog,
) -> tuple[list[GeneratedPolicy], list[GeneratedPolicy]]:
    """Safety net for the recurring failure mode where PGN REGISTERS fine but FAILS TO COMPILE at enforcement
    — e.g. a constraint on a raw list column (cc/bcc) or a `contains()` UDF. Passant then raises a rewrite
    error that fail-closes and blocks the ENTIRE sink (every call), silently wrecking utility. We probe each
    policy through the real validate path in a scratch DuckDB and drop any whose enforcement raises a
    rewrite/compile error. Purely structural (no schema- or suite-specific knowledge): keep only what actually
    compiles. Cheap (~tens of ms/task; a rebuild only on the rare drop)."""
    rewrite_signs = ("rewriteerror", "unsupported query form", "tuple udf", "non-empty select projection")
    try:
        import duckdb
        from data_flow_control import Policy, dfc

        from dfc_agent_framework_integration.events import create_event_tables
        from dfc_agent_framework_integration.materialize import materialize_preamble_data
        from dfc_agent_framework_integration.runtime import DFCRuntimeValidator
    except Exception:
        return policies, []

    tool_by_relation = {r.name: r.tool_name for r in runtime_schema.tool_input_relations}

    def is_rewrite_error(text: str) -> bool:
        low = (text or "").lower()
        return any(sign in low for sign in rewrite_signs)

    def build():
        raw = duckdb.connect()
        create_event_tables(raw, runtime_schema)
        materialize_preamble_data(raw, preamble_facts, runtime_schema.preamble_relation)
        return raw, dfc(raw)

    try:
        raw, conn = build()
    except Exception:
        return policies, []  # cannot build the probe env -> do not filter (fail open: registration handles the rest)

    good: list[GeneratedPolicy] = []
    dropped: list[GeneratedPolicy] = []
    for policy in policies:
        tool = tool_by_relation.get(policy.applies_to_relation)
        broken = False
        try:
            conn.register_policy(Policy.from_pgn(policy.pgn))
            conn.refresh_catalog()
            if tool is not None:
                validator = DFCRuntimeValidator(
                    raw, conn, runtime_schema, good + [policy],
                    [p.policy_id for p in good] + [policy.policy_id], preamble_facts)
                violation = validator.validate_tool_call(tool, {})  # NULL payload; only a COMPILE error matters here
                if violation is not None and is_rewrite_error(str(violation.raw_error)):
                    broken = True
        except Exception as exc:  # a non-rewrite registration/parse error is the repair loop's job — keep it
            broken = is_rewrite_error(str(exc))
        if broken:
            event_log.log("policy_dropped_uncompilable", policy_id=policy.policy_id, sink=policy.applies_to_relation)
            dropped.append(policy)
            try:  # rebuild the probe env without the bad policy so later checks aren't polluted
                raw, conn = build()
                for kept in good:
                    conn.register_policy(Policy.from_pgn(kept.pgn))
                conn.refresh_catalog()
            except Exception:
                break
        else:
            good.append(policy)
    return good, dropped


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
    max_feedback_rounds: int = 2,
) -> tuple[list[GeneratedPolicy], list[str], list[DeletedPolicyRecord], list[PolicyRegistrationRecord]]:
    event_log = event_log or DFCEventLog(None)
    generated = llm.parse(
        model=model,
        instructions=policy_generation_instructions(runtime_schema, preamble_facts),
        input_text=preamble,
        text_format=GeneratedPolicySet,
    ).policies
    event_log.log("policy_generation_llm_complete", policy_count=len(generated))

    # The Layer-2 classifier is per-tool, so its verdict is identical every round — classify once, reuse.
    read_only_sinks = _classify_read_only_sinks(
        llm, model=classifier_model or model, runtime_schema=runtime_schema, event_log=event_log
    )
    tool_by_relation = {r.name: r.tool_name for r in runtime_schema.tool_input_relations}

    # Bounded feedback loop: admit (Layer 2/3) and, when enabled, probe each admitted grounding policy.
    # The probe (allow-legit / block-synthetic) re-registers each policy in a scratch DuckDB — CPU-bound
    # and ~15-20s/task, catching only the rare malformed/vacuous cases that admission control + Layer 3
    # already mostly cover. It is OFF by default for speed; set DFC_ENABLE_PROBE=1 for the thorough path.
    # Either way, the loop still feeds admission-control drops back to the generator and revises.
    probe_enabled = os.getenv("DFC_ENABLE_PROBE", "").lower() in ("1", "true", "yes")
    admitted: list[GeneratedPolicy] = []
    false_positive_ids: set[str] = set()
    for round_idx in range(max_feedback_rounds + 1):
        admitted, problems = _admit(generated, read_only_sinks, preamble_facts, event_log)
        false_positive_ids = set()
        if probe_enabled:
            for policy in admitted:
                probe = _probe_one_policy(runtime_schema, preamble_facts, policy, tool_by_relation)
                if probe["false_positive"]:
                    false_positive_ids.add(policy.policy_id)
                    event_log.log("policy_probe_false_positive", policy_id=policy.policy_id, sink=policy.applies_to_relation)
                    problems.append({"policy_id": policy.policy_id, "sink": policy.applies_to_relation, "reason": "false_positive",
                                     "detail": "BLOCKS the task's own legitimate value; re-ground so the authorized value passes, or remove"})
                elif probe["vacuous"]:
                    event_log.log("policy_probe_vacuous", policy_id=policy.policy_id, sink=policy.applies_to_relation)
                    problems.append({"policy_id": policy.policy_id, "sink": policy.applies_to_relation, "reason": "vacuous",
                                     "detail": "does not constrain the sink (allows any value); tighten or remove"})
        if not problems or round_idx == max_feedback_rounds:
            if problems:
                event_log.log("policy_feedback_exhausted", round=round_idx, problem_count=len(problems))
            break
        event_log.log("policy_feedback_round", round=round_idx + 1,
                      reasons=sorted({p["reason"] for p in problems}), problem_count=len(problems))
        generated = llm.parse(
            model=model,
            instructions=revise_policies_instructions(runtime_schema, preamble_facts, generated, problems),
            input_text=preamble,
            text_format=GeneratedPolicySet,
        ).policies

    # Drop confirmed false positives feedback could not fix — a policy that provably blocks the task's own
    # value is a guaranteed utility break, not a real guard.
    if false_positive_ids:
        event_log.log("policy_dropped_false_positive_final", policy_ids=sorted(false_positive_ids))
        admitted = [p for p in admitted if p.policy_id not in false_positive_ids]

    # Final structural safety net: drop policies that register but DON'T COMPILE at enforcement (they would
    # fail-closed and block their whole sink). Working destination guards on the same sink survive.
    admitted, uncompilable = _drop_uncompilable_policies(admitted, runtime_schema, preamble_facts, event_log)
    if uncompilable:
        event_log.log("policy_dropped_uncompilable_final", policy_ids=sorted(p.policy_id for p in uncompilable))

    # Scope default-deny to recipient/account sinks: drop any DEFAULT-DENY on a URL-visit sink (browse / web-form
    # / download). Blanket-blocking URLs breaks legitimate link-following (e.g. dailylife "follow the link X
    # emailed"); off-task URLs are vetted by named-URL allowlists + the semantic judge, not a blanket deny. Real
    # url ALLOWLISTS (grounded on a genuine authorized_url fact) are unaffected.
    url_deny_ids = {p.policy_id for p in admitted if _is_url_default_deny(p)}
    if url_deny_ids:
        admitted = [p for p in admitted if p.policy_id not in url_deny_ids]
        event_log.log("policy_dropped_url_default_deny", policy_ids=sorted(url_deny_ids))

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
