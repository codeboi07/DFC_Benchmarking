from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class BenchmarkTool(Protocol):
    name: str
    description: str
    parameters: type[BaseModel]
    return_type: Any | None
