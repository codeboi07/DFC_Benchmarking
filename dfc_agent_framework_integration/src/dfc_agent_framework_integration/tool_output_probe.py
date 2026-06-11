from __future__ import annotations

import re
from typing import Any, get_origin

from dfc_agent_framework_integration.tools import BenchmarkTool

_SCALAR_DICT_VALUE_TYPES = (str, int, float, bool, type(None))
_SQL_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def return_type_is_dict(return_type: Any) -> bool:
    if return_type is dict:
        return True
    return get_origin(return_type) is dict


def normalize_dict_key_to_column(key: str, *, used: set[str] | None = None) -> str:
    used = used or set()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", key.strip().lower()).strip("_")
    if not normalized:
        normalized = "field"
    if not _SQL_IDENTIFIER.match(normalized):
        normalized = f"field_{normalized}"
    candidate = normalized
    suffix = 2
    while candidate in used:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def is_flat_scalar_dict(value: dict[Any, Any]) -> bool:
    if not value:
        return False
    return all(isinstance(item, _SCALAR_DICT_VALUE_TYPES) for item in value.values())


def mock_value_for_property(property_schema: dict[str, Any], field_name: str) -> Any:
    if "default" in property_schema:
        return property_schema["default"]
    if "anyOf" in property_schema:
        for option in property_schema["anyOf"]:
            if isinstance(option, dict) and option.get("type") == "null":
                continue
            if isinstance(option, dict):
                return mock_value_for_property(option, field_name)
    property_type = property_schema.get("type")
    if property_type == "string":
        return f"dfc_probe_{field_name}"
    if property_type == "integer":
        return 1
    if property_type == "number":
        return 1.0
    if property_type == "boolean":
        return True
    if property_type == "array":
        items = property_schema.get("items", {})
        if isinstance(items, dict) and items.get("type") == "string":
            return [f"dfc_probe_{field_name}"]
        return []
    if property_type == "object":
        return {}
    return f"dfc_probe_{field_name}"


def mock_arguments_for_function(function: BenchmarkTool) -> dict[str, Any]:
    schema = function.parameters.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    mocked: dict[str, Any] = {}
    for field_name, property_schema in properties.items():
        prop = property_schema if isinstance(property_schema, dict) else {}
        if field_name in required or not required:
            mocked[field_name] = mock_value_for_property(prop, field_name)
    return mocked


def columns_from_flat_dict(
    sample: dict[Any, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    columns: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    source_key_by_column: dict[str, str] = {}
    used: set[str] = set()
    for key, value in sample.items():
        key_text = str(key)
        column = normalize_dict_key_to_column(key_text, used=used)
        columns[column] = "VARCHAR"
        descriptions[column] = f"Value of {key_text!r} from tool output."
        source_key_by_column[column] = key_text
    return columns, descriptions, source_key_by_column


def probe_dict_output_schema(
    function: BenchmarkTool,
    *,
    functions_runtime: Any,
    env: Any,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]] | None:
    if not return_type_is_dict(function.return_type):
        return None

    mock_args = mock_arguments_for_function(function)
    try:
        result, error = functions_runtime.run_function(env, function.name, mock_args)
    except Exception:
        return None
    if error is not None or not isinstance(result, dict) or not is_flat_scalar_dict(result):
        return None
    return columns_from_flat_dict(result)
