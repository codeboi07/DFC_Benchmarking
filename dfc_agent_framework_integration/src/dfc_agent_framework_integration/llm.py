from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient(Protocol):
    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        text_format: type[T],
    ) -> T: ...


class OpenAIStructuredLLMClient:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def parse(
        self,
        *,
        model: str | None = None,
        instructions: str,
        input_text: str,
        text_format: type[T],
    ) -> T:
        response = self._client.responses.parse(
            model=model or self._model,
            instructions=instructions,
            input=input_text,
            text_format=text_format,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError(f"Structured output parsing failed for {text_format.__name__}")
        return parsed


class FakeStructuredLLMClient:
    def __init__(self, responses: dict[type[BaseModel], list[BaseModel]] | None = None) -> None:
        self._responses: dict[type[BaseModel], list[BaseModel]] = responses or {}
        self.calls: list[dict[str, Any]] = []

    def register(self, text_format: type[BaseModel], responses: list[BaseModel]) -> None:
        self._responses[text_format] = list(responses)

    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        text_format: type[T],
    ) -> T:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "input_text": input_text,
                "text_format": text_format,
            }
        )
        queue = self._responses.get(text_format, [])
        if not queue:
            raise ValueError(f"No fake response registered for {text_format.__name__}")
        item = queue.pop(0)
        if not isinstance(item, text_format):
            raise TypeError(f"Fake response type mismatch: expected {text_format}, got {type(item)}")
        return item
