from __future__ import annotations

import pytest

from dfc_agent_framework_integration.llm import FakeStructuredLLMClient
from dfc_agent_framework_integration.schema import (
    GeneratedPolicy,
    GeneratedPolicySet,
    PolicyRepairDecision,
    PreambleExtraction,
)
from agentdojo.functions_runtime import FunctionsRuntime, make_function


def send_email(recipients: list[str], subject: str, body: str) -> str:
    """Send an email.

    :param recipients: recipient email addresses
    :param subject: email subject
    :param body: email body
    """
    return f"sent:{','.join(recipients)}"


@pytest.fixture
def send_email_function():
    return make_function(send_email)


@pytest.fixture
def send_email_runtime(send_email_function):
    return FunctionsRuntime([send_email_function])


@pytest.fixture
def standard_extraction_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [
            PreambleExtraction(
                facts={
                    "authorized_recipient_email": "alice@example.com",
                }
            )
        ],
    )
    return llm


@pytest.fixture
def standard_policy_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(
        GeneratedPolicySet,
        [
            GeneratedPolicySet(
                policies=[
                    GeneratedPolicy(
                        policy_id="send_email_recipient",
                        pgn=(
                            "SINK SendEmailInput\n"
                            "DIMENSION PreambleData\n"
                            "CONSTRAINT max(SendEmailInput.recipient) = PreambleData.authorized_recipient_email\n"
                            "ON FAIL KILL\n"
                            "DESCRIPTION Only send email to the recipient authorized by the original task preamble."
                        ),
                        description="Only send email to the recipient authorized by the original task preamble.",
                        applies_to_relation="SendEmailInput",
                        applies_to_event="tool_call:send_email",
                        rationale="Ground recipient to preamble.",
                    )
                ]
            )
        ],
    )
    return llm


@pytest.fixture
def combined_llm(standard_extraction_llm, standard_policy_llm) -> FakeStructuredLLMClient:
    combined = FakeStructuredLLMClient()
    for text_format, responses in standard_extraction_llm._responses.items():
        combined.register(text_format, list(responses))
    for text_format, responses in standard_policy_llm._responses.items():
        combined.register(text_format, list(responses))
    return combined


@pytest.fixture
def repair_delete_llm() -> FakeStructuredLLMClient:
    llm = FakeStructuredLLMClient()
    llm.register(
        PolicyRepairDecision,
        [
            PolicyRepairDecision(
                delete=True,
                rationale="Cannot express policy with available columns.",
            )
        ],
    )
    return llm
