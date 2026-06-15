from __future__ import annotations

import re

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import EXTRACTION_INSTRUCTIONS
from dfc_agent_framework_integration.schema import PreambleExtraction, PreambleFact

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_PREFIXES = ("__dfc_", "__passant_", "dfc_")
MAX_KEY_LENGTH = 128
MAX_VALUE_LENGTH = 4096


class ExtractionValidationError(ValueError):
    pass


class ExtractionError(Exception):
    pass


def facts_to_dict(facts: list[PreambleFact]) -> dict[str, str]:
    result: dict[str, str] = {}
    for fact in facts:
        if fact.key in result:
            raise ExtractionValidationError(f"Duplicate key {fact.key!r}")
        result[fact.key] = fact.value
    return result


def validate_extraction_facts(facts: dict[str, str]) -> None:
    if not facts:
        raise ExtractionValidationError("At least one fact must be extracted")
    for key, value in facts.items():
        if not _KEY_PATTERN.match(key):
            raise ExtractionValidationError(f"Key {key!r} must match ^[a-z][a-z0-9_]*$")
        if any(key.startswith(prefix) for prefix in _RESERVED_PREFIXES):
            raise ExtractionValidationError(f"Key {key!r} uses a reserved prefix")
        if len(key) > MAX_KEY_LENGTH:
            raise ExtractionValidationError(f"Key {key!r} exceeds maximum length")
        if not isinstance(value, str):
            raise ExtractionValidationError(f"Value for {key!r} must be a string")
        if len(value) > MAX_VALUE_LENGTH:
            raise ExtractionValidationError(f"Value for {key!r} exceeds maximum length")


def extract_preamble_facts(
    llm: StructuredLLMClient,
    *,
    model: str,
    preamble: str,
    allow_repair: bool = True,
    event_log: DFCEventLog | None = None,
) -> dict[str, str]:
    event_log = event_log or DFCEventLog(None)
    extraction = _call_extraction_llm(llm, model=model, preamble=preamble)
    try:
        facts = facts_to_dict(extraction.facts)
        validate_extraction_facts(facts)
        event_log.log("extraction_complete", fact_keys=sorted(facts.keys()))
        return facts
    except ExtractionValidationError as first_error:
        if not allow_repair:
            raise ExtractionError(str(first_error)) from first_error
        event_log.log("extraction_repair_start", error=str(first_error))
        repair_input = (
            f"{preamble}\n\nPrevious extraction failed validation: {first_error}\nReturn corrected facts only."
        )
        repaired = _call_extraction_llm(llm, model=model, preamble=repair_input)
        try:
            repaired_facts = facts_to_dict(repaired.facts)
            validate_extraction_facts(repaired_facts)
            event_log.log("extraction_repair_complete", fact_keys=sorted(repaired_facts.keys()))
            return repaired_facts
        except ExtractionValidationError as second_error:
            raise ExtractionError(str(second_error)) from second_error


def _call_extraction_llm(
    llm: StructuredLLMClient,
    *,
    model: str,
    preamble: str,
) -> PreambleExtraction:
    return llm.parse(
        model=model,
        instructions=EXTRACTION_INSTRUCTIONS,
        input_text=preamble,
        text_format=PreambleExtraction,
    )
