from __future__ import annotations

import pytest

from dfc_agent_framework_integration.extraction import ExtractionError, extract_preamble_facts, validate_extraction_facts
from dfc_agent_framework_integration.llm import FakeStructuredLLMClient
from dfc_agent_framework_integration.schema import PreambleExtraction


def test_validate_extraction_rejects_invalid_keys():
    with pytest.raises(Exception):
        validate_extraction_facts({"BadKey": "value"})


def test_extract_preamble_facts_returns_structured_facts(standard_extraction_llm: FakeStructuredLLMClient):
    facts = extract_preamble_facts(
        standard_extraction_llm,
        model="fake-model",
        preamble="Send an email to alice@example.com about the invoice.",
    )
    assert facts == {"authorized_recipient_email": "alice@example.com"}


def test_invalid_extraction_keys_trigger_one_repair_retry():
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [
            PreambleExtraction.from_dict({"BadKey": "alice@example.com"}),
            PreambleExtraction.from_dict({"authorized_recipient_email": "alice@example.com"}),
        ],
    )
    facts = extract_preamble_facts(llm, model="fake-model", preamble="task")
    assert facts["authorized_recipient_email"] == "alice@example.com"
    assert len([call for call in llm.calls if call["text_format"] is PreambleExtraction]) == 2


def test_extraction_fails_closed_after_repair():
    llm = FakeStructuredLLMClient()
    llm.register(
        PreambleExtraction,
        [
            PreambleExtraction.from_dict({"BadKey": "value"}),
            PreambleExtraction.from_dict({"AlsoBad": "value"}),
        ],
    )
    with pytest.raises(ExtractionError):
        extract_preamble_facts(llm, model="fake-model", preamble="task")
