from __future__ import annotations

from typing import Any

import duckdb
from data_flow_control import dfc

from dfc_agent_framework_integration.events import (
    ASSISTANT_RESPONSE_RELATION,
    PROMPT_INPUT_RELATION,
    create_event_tables,
)
from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
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
        dfc_model: str,
        agent_model: str | None = None,
        event_log: DFCEventLog | None = None,
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
        self.dfc_model = dfc_model
        self.agent_model = agent_model
        self.event_log = event_log or DFCEventLog(None)
        self._closed = False

    @property
    def model(self) -> str:
        """Backward-compatible alias for the DFC LLM model."""
        return self.dfc_model

    @classmethod
    def prepare_task(
        cls,
        *,
        task_context: BenchmarkTaskContext,
        runtime_schema: RuntimeSchema,
        llm: Any,
        dfc_model: str,
        functions: dict[str, Any],
        agent_model: str | None = None,
        classifier_model: str | None = None,
        event_log: DFCEventLog | None = None,
    ) -> DFCBenchmarkContext:
        event_log = event_log or DFCEventLog(None)
        event_log.log(
            "prepare_task_start",
            task_id=task_context.task_id,
            dfc_model=dfc_model,
            agent_model=agent_model,
            preamble_preview=task_context.preamble[:200],
        )

        raw_conn = duckdb.connect()
        create_event_tables(raw_conn, runtime_schema)
        event_log.log(
            "schema_tables_created",
            relation_count=len(runtime_schema.relation_names()),
        )

        event_log.log("extraction_start", model=dfc_model)
        extracted_facts = extract_preamble_facts(
            llm,
            model=dfc_model,
            preamble=task_context.preamble,
            event_log=event_log,
        )
        materialize_preamble_data(raw_conn, extracted_facts, runtime_schema.preamble_relation)
        event_log.log(
            "preamble_materialized",
            fact_keys=sorted(extracted_facts.keys()),
        )

        dfc_conn = dfc(raw_conn)
        event_log.log("policy_generation_start", model=dfc_model)
        generated_policies, registered_policy_ids, deleted, policy_registration = generate_and_register_policies(
            dfc_conn,
            llm,
            model=dfc_model,
            preamble=task_context.preamble,
            preamble_facts=extracted_facts,
            runtime_schema=runtime_schema,
            event_log=event_log,
            classifier_model=classifier_model,
        )
        dfc_conn.refresh_catalog()
        event_log.log(
            "policy_generation_complete",
            generated_count=len(generated_policies),
            registered_policy_ids=registered_policy_ids,
            deleted_count=len(deleted),
            total_repair_attempts=sum(record.repair_attempts for record in policy_registration),
        )

        diagnostics = DFCTaskDiagnostics(
            extracted_facts=extracted_facts,
            generated_policies=generated_policies,
            registered_policy_ids=registered_policy_ids,
            deleted_policies=deleted,
            policy_registration=policy_registration,
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
        context = cls(
            raw_conn=raw_conn,
            conn=dfc_conn,
            task_context=task_context,
            runtime_schema=runtime_schema,
            extracted_facts=extracted_facts,
            generated_policies=generated_policies,
            registered_policy_ids=registered_policy_ids,
            diagnostics=diagnostics,
            validator=validator,
            dfc_model=dfc_model,
            agent_model=agent_model,
            event_log=event_log,
        )
        event_log.log(
            "prepare_task_complete",
            registered_policy_ids=registered_policy_ids,
        )
        return context

    def _record_validation_event(self, event_type: str, violation: Any) -> None:
        if violation is not None:
            for policy_id in violation.policy_ids:
                self.diagnostics.policy_fire_counts[policy_id] = (
                    self.diagnostics.policy_fire_counts.get(policy_id, 0) + 1
                )
        self.diagnostics.validation_events.append(
            {
                "event_type": event_type,
                "blocked": violation is not None,
                "policy_ids": violation.policy_ids if violation is not None else [],
                "policy_descriptions": violation.policy_descriptions if violation else [],
            }
        )
        self.event_log.log(
            "validation",
            event_type=event_type,
            blocked=violation is not None,
            policy_ids=violation.policy_ids if violation is not None else [],
            policy_descriptions=violation.policy_descriptions if violation else [],
            attempted_payload=getattr(violation, "attempted_payload", None) if violation is not None else None,
        )

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
        violation = self.validator.validate_tool_call(tool_name, args)
        self._record_validation_event(f"tool_call:{tool_name}", violation)
        return violation

    def validate_event(self, *, event_type: str, relation: str, payload: dict[str, Any]) -> Any:
        violation = self.validator.validate_event(
            event_type=event_type,
            relation=relation,
            payload=payload,
        )
        self._record_validation_event(event_type, violation)
        return violation

    def record_tool_output(self, tool_name: str, result: Any) -> None:
        self.validator.record_tool_output(tool_name, result)
        self.event_log.log("tool_output_recorded", tool_name=tool_name)

    def validate_assistant_response(self, content: str) -> Any:
        return self.validate_event(
            event_type="assistant_response",
            relation=ASSISTANT_RESPONSE_RELATION,
            payload={"content": content},
        )

    def validate_prompt(self, content: str, target: str = "") -> Any:
        return self.validate_event(
            event_type="prompt",
            relation=PROMPT_INPUT_RELATION,
            payload={"content": content, "target": target},
        )

    def close(self) -> None:
        if self._closed:
            return
        self.event_log.log("context_closed")
        self.conn.close()
        self._closed = True
