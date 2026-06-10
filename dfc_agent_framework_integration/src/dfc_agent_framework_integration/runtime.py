from __future__ import annotations

from typing import Any

from dfc_agent_framework_integration.events import (
    ASSISTANT_RESPONSE_RELATION,
    PROMPT_INPUT_RELATION,
    attempt_write_event_row,
    build_insert_row,
    clear_relation_rows,
    expand_payload_rows,
    input_table_name,
    new_event_id,
    quote_identifier,
)
from dfc_agent_framework_integration.schema import DFCViolation, GeneratedPolicy, RuntimeSchema
from dfc_agent_framework_integration.validation import identify_violated_policy_descriptions


class DFCRuntimeValidator:
    def __init__(
        self,
        raw_conn: Any,
        dfc_conn: Any,
        runtime_schema: RuntimeSchema,
        generated_policies: list[GeneratedPolicy],
        registered_policy_ids: list[str],
        preamble_facts: dict[str, str],
        task_id: str | None = None,
    ) -> None:
        self.raw_conn = raw_conn
        self.dfc_conn = dfc_conn
        self.runtime_schema = runtime_schema
        self.generated_policies = generated_policies
        self.registered_policy_ids = registered_policy_ids
        self.preamble_facts = preamble_facts
        self.task_id = task_id
        self._relation_columns = {
            relation.name: relation.columns for relation in runtime_schema.all_relations()
        }

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> DFCViolation | None:
        relation_name = input_table_name(tool_name)
        if relation_name not in self._relation_columns:
            return None
        return self.validate_event(
            event_type=f"tool_call:{tool_name}",
            relation=relation_name,
            payload=args,
        )

    def validate_event(
        self,
        *,
        event_type: str,
        relation: str,
        payload: dict[str, Any],
    ) -> DFCViolation | None:
        if relation not in self._relation_columns:
            raise ValueError(f"Unknown relation {relation!r}")

        relation_columns = self._relation_columns[relation]
        payload_rows = expand_payload_rows(payload, relation_columns)
        clear_relation_rows(self.raw_conn, relation)
        event_id = new_event_id()
        blocked_row: dict[str, Any] | None = None
        raw_error: str | None = None

        for index, payload_row in enumerate(payload_rows):
            row = build_insert_row(
                payload_row,
                relation_columns,
                event_id=f"{event_id}:{index}",
                task_id=self.task_id,
                event_type=event_type,
                status="pending",
            )
            allowed, error = attempt_write_event_row(
                self.dfc_conn,
                self.raw_conn,
                relation,
                row,
                relation_columns,
            )
            if allowed:
                self.raw_conn.execute(
                    f"UPDATE {quote_identifier(relation)} SET {quote_identifier('__dfc_status')} = ? "
                    f"WHERE {quote_identifier('__dfc_event_id')} = ?",
                    ["allowed", f"{event_id}:{index}"],
                )
                continue

            blocked_row = row
            raw_error = error
            break

        if blocked_row is None:
            return None

        descriptions = identify_violated_policy_descriptions(
            relation_name=relation,
            attempted_row=blocked_row,
            relation_columns=relation_columns,
            generated_policies=self.generated_policies,
            registered_policy_ids=self.registered_policy_ids,
            preamble_facts=self.preamble_facts,
            raw_error=raw_error,
        )
        return DFCViolation(
            event_type=event_type,
            relation=relation,
            attempted_payload=payload,
            policy_descriptions=descriptions,
            raw_error=raw_error,
        )

    def validate_assistant_response(self, content: str) -> DFCViolation | None:
        return self.validate_event(
            event_type="assistant_response",
            relation=ASSISTANT_RESPONSE_RELATION,
            payload={"content": content},
        )

    def validate_prompt(self, content: str, target: str = "") -> DFCViolation | None:
        return self.validate_event(
            event_type="prompt",
            relation=PROMPT_INPUT_RELATION,
            payload={"content": content, "target": target},
        )


def format_violation_message(violation: DFCViolation, tool_name: str | None = None) -> str:
    policy_text = violation.policy_descriptions[0] if violation.policy_descriptions else "Data flow policy violation."
    lines = [
        "Data flow policy violation. The tool call was not executed.",
        f"Policy: {policy_text}",
    ]
    if tool_name:
        lines.append(f"Attempted tool: {tool_name}")
    lines.append(f"Attempted arguments: {violation.attempted_payload}")
    lines.append("Try again using only values authorized by the original task.")
    return "\n".join(lines)


def format_response_violation_message(violation: DFCViolation) -> str:
    policy_text = violation.policy_descriptions[0] if violation.policy_descriptions else "Data flow policy violation."
    return (
        "Data flow policy violation. Your response was not accepted.\n"
        f"Policy: {policy_text}\n"
        "Revise your response using only values authorized by the original task."
    )


DFC_VIOLATION_USER_PREFIX = "Data flow policy violation."


def is_dfc_feedback_user_message(content: str) -> bool:
    return content.strip().startswith(DFC_VIOLATION_USER_PREFIX)


def format_prompt_violation_message(violation: DFCViolation) -> str:
    policy_text = violation.policy_descriptions[0] if violation.policy_descriptions else "Data flow policy violation."
    return (
        f"{DFC_VIOLATION_USER_PREFIX} Revise the user request before continuing.\n"
        f"Policy: {policy_text}\n"
        "Try again using only values authorized by the original task."
    )
