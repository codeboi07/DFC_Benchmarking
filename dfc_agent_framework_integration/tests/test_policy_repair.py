from __future__ import annotations

from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.llm import FakeStructuredLLMClient
from dfc_agent_framework_integration.schema import (
    BenchmarkTaskContext,
    GeneratedPolicy,
    GeneratedPolicySet,
    PolicyRepairDecision,
    PreambleExtraction,
    RuntimeSchema,
)


def _build_context_with_policies(llm: FakeStructuredLLMClient, policies: list[GeneratedPolicy], send_email_runtime):
    return DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=llm,
        model="fake-model",
        functions=send_email_runtime.functions,
    )


def test_invalid_pgn_triggers_repair(send_email_runtime):
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="bad_policy",
                        pgn="SINK SendEmailInput CONSTRAINT broken ON FAIL KILL",
                        description="broken",
                        applies_to_relation="SendEmailInput",
                        applies_to_event="tool_call:send_email",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    llm.register(
        PolicyRepairDecision,
        [
            PolicyRepairDecision(
                delete=False,
                repaired_pgn=(
                    "SINK SendEmailInput\n"
                    "DIMENSION PreambleData\n"
                    "CONSTRAINT max(SendEmailInput.recipient) = PreambleData.authorized_recipient_email\n"
                    "ON FAIL KILL\n"
                    "DESCRIPTION repaired policy"
                ),
                repaired_description="repaired policy",
                rationale="fixed syntax",
            )
        ],
    )
    context = _build_context_with_policies(llm, [], send_email_runtime)
    try:
        assert "bad_policy" in context.registered_policy_ids
        assert any(call["text_format"] is PolicyRepairDecision for call in llm.calls)
    finally:
        context.close()


def test_runtime_probe_deletes_ilike_policies(send_email_runtime):
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="bad_runtime_policy",
                        pgn=(
                            "SINK SendEmailInput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(SendEmailInput.recipient) ILIKE "
                            "'%' || max(PreambleData.authorized_recipient_email) || '%'\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION bad runtime policy"
                        ),
                        description="bad runtime policy",
                        applies_to_relation="SendEmailInput",
                        applies_to_event="tool_call:send_email",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    llm.register(
        PolicyRepairDecision,
        [
            PolicyRepairDecision(
                delete=True,
                repaired_pgn=None,
                repaired_description=None,
                rationale="ILIKE with aggregates is unsupported at runtime.",
            )
        ],
    )

    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=llm,
        model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        assert context.registered_policy_ids == []
        assert len(context.diagnostics.deleted_policies) == 1
    finally:
        context.close()


def test_delete_true_drops_only_failed_policy(send_email_runtime, repair_delete_llm):
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"})],
    )
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="bad_policy",
                        pgn="SOURCE MissingTable CONSTRAINT x = 1 ON FAIL KILL",
                        description="bad",
                        applies_to_relation="SendEmailInput",
                        applies_to_event="tool_call:send_email",
                        rationale="test",
                    )
                ]
            )
        ],
    )
    llm.register(PolicyRepairDecision, list(repair_delete_llm._responses[PolicyRepairDecision]))

    context = _build_context_with_policies(llm, [], send_email_runtime)
    try:
        assert context.registered_policy_ids == []
        assert len(context.diagnostics.deleted_policies) == 1
        assert context.diagnostics.deleted_policies[0].policy_id == "bad_policy"
    finally:
        context.close()
