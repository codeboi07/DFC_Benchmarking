from __future__ import annotations

import re

from dfc_agent_framework_integration.llm import StructuredLLMClient
from dfc_agent_framework_integration.prompts import EXTRACTION_INSTRUCTIONS
from dfc_agent_framework_integration.schema import PreambleExtraction

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_PREFIXES = ("__dfc_", "__passant_", "dfc_")
MAX_KEY_LENGTH = 128
MAX_VALUE_LENGTH = 4096


class ExtractionValidationError(ValueError):
    pass


class ExtractionError(Exception):
    pass


def validate_extraction_facts(facts: dict[str, str]) -> None:
    if not facts:
        raise ExtractionValidationError("At least one fact must be extracted")
    for key, value in facts.items():
        if not _KEY_PATTERN.match(key):
            raise ExtractionValidationError(
                f"Key {key!r} must match ^[a-z][a-z0-9_]*$"
            )
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
) -> dict[str, str]:
    extraction = _call_extraction_llm(llm, model=model, preamble=preamble)
    try:
        validate_extraction_facts(extraction.facts)
        return extraction.facts
    except ExtractionValidationError as first_error:
        if not allow_repair:
            raise ExtractionError(str(first_error)) from first_error
        repair_input = (
            f"{preamble}\n\n"
            f"Previous extraction failed validation: {first_error}\n"
            "Return corrected facts only."
        )
        repaired = _call_extraction_llm(llm, model=model, preamble=repair_input)
        try:
            validate_extraction_facts(repaired.facts)
            return repaired.facts
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
