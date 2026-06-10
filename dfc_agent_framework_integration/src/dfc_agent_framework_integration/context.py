from __future__ import annotations

from typing import Any

import duckdb
from data_flow_control import dfc

from dfc_agent_framework_integration.events import create_event_tables
from dfc_agent_framework_integration.extraction import extract_preamble_facts
from dfc_agent_framework_integration.materialize import materialize_preamble_data
from dfc_agent_framework_integration.policies import generate_and_register_policies
from dfc_agent_framework_integration.runtime import DFCRuntimeValidator
from dfc_agent_framework_integration.schema import BenchmarkTaskContext, DFCTaskDiagnostics, RuntimeSchema


class DFCBenchmarkContext:
    def __init__(
        self,
        *,
        raw_conn: Any,
        conn: Any,
        task_context: BenchmarkTaskContext,
        runtime_schema: RuntimeSchema,
        extracted_facts: dict[str, str],
        generated_policies: list[Any],
        registered_policy_ids: list[str],
        diagnostics: DFCTaskDiagnostics,
        validator: DFCRuntimeValidator,
        model: str,
    ) -> None:
        self.raw_conn = raw_conn
        self.conn = conn
        self.task_context = task_context
        self.runtime_schema = runtime_schema
        self.extracted_facts = extracted_facts
        self.generated_policies = generated_policies
        self.registered_policy_ids = registered_policy_ids
        self.diagnostics = diagnostics
        self.validator = validator
        self.model = model
        self._closed = False

    @classmethod
    def prepare_task(
        cls,
        *,
        task_context: BenchmarkTaskContext,
        runtime_schema: RuntimeSchema,
        llm: Any,
        model: str,
        functions: dict[str, Any],
    ) -> DFCBenchmarkContext:
        raw_conn = duckdb.connect()
        create_event_tables(raw_conn, runtime_schema)

        extracted_facts = extract_preamble_facts(
            llm,
            model=model,
            preamble=task_context.preamble,
        )
        materialize_preamble_data(raw_conn, extracted_facts, runtime_schema.preamble_relation)

        dfc_conn = dfc(raw_conn)
        generated_policies, registered_policy_ids, deleted = generate_and_register_policies(
            dfc_conn,
            llm,
            model=model,
            preamble=task_context.preamble,
            preamble_facts=extracted_facts,
            runtime_schema=runtime_schema,
            raw_conn=raw_conn,
        )
        dfc_conn.refresh_catalog()

        diagnostics = DFCTaskDiagnostics(
            extracted_facts=extracted_facts,
            generated_policies=generated_policies,
            registered_policy_ids=registered_policy_ids,
            deleted_policies=deleted,
        )
        validator = DFCRuntimeValidator(
            raw_conn,
            dfc_conn,
            runtime_schema,
            generated_policies,
            registered_policy_ids,
            extracted_facts,
            task_id=task_context.task_id,
        )
        return cls(
            raw_conn=raw_conn,
            conn=dfc_conn,
            task_context=task_context,
            runtime_schema=runtime_schema,
            extracted_facts=extracted_facts,
            generated_policies=generated_policies,
            registered_policy_ids=registered_policy_ids,
            diagnostics=diagnostics,
            validator=validator,
            model=model,
        )

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
        violation = self.validator.validate_tool_call(tool_name, args)
        self.diagnostics.validation_events.append(
            {
                "event_type": f"tool_call:{tool_name}",
                "blocked": violation is not None,
                "policy_descriptions": violation.policy_descriptions if violation else [],
            }
        )
        return violation

    def validate_event(self, *, event_type: str, relation: str, payload: dict[str, Any]) -> Any:
        violation = self.validator.validate_event(
            event_type=event_type,
            relation=relation,
            payload=payload,
        )
        self.diagnostics.validation_events.append(
            {
                "event_type": event_type,
                "blocked": violation is not None,
                "policy_descriptions": violation.policy_descriptions if violation else [],
            }
        )
        return violation

    def validate_assistant_response(self, content: str) -> Any:
        return self.validate_event(
            event_type="assistant_response",
            relation="AssistantResponseOutput",
            payload={"content": content},
        )

    def validate_prompt(self, content: str, target: str = "") -> Any:
        return self.validate_event(
            event_type="prompt",
            relation="PromptInput",
            payload={"content": content, "target": target},
        )

    def close(self) -> None:
        if self._closed:
            return
        self.conn.close()
        self._closed = True
