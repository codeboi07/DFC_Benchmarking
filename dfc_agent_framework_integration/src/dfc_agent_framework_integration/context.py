from __future__ import annotations

from typing import Any

import duckdb
from data_flow_control import dfc

from dfc_agent_framework_integration import semantic_judge
from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.events import (
    ASSISTANT_RESPONSE_RELATION,
    PROMPT_INPUT_RELATION,
    create_event_tables,
    quote_identifier,
)
from dfc_agent_framework_integration.extraction import extract_preamble_facts
from dfc_agent_framework_integration.materialize import materialize_preamble_data
from dfc_agent_framework_integration.policies import generate_and_register_policies
from dfc_agent_framework_integration.runtime import DFCRuntimeValidator
from dfc_agent_framework_integration.schema import (
    BenchmarkTaskContext,
    DFCTaskDiagnostics,
    RuntimeSchema,
    SourceRequiredResult,
    SourceRequiredSink,
)
from dfc_agent_framework_integration.source_required import (
    build_source_required_index,
    validate_source_required_sql,
)


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _format_rows_for_judge(rows: list[dict[str, Any]], limit: int = 25) -> str:
    if not rows:
        return "(no rows)"
    return "\n".join(
        "; ".join(f"{key}={str(value)[:300]}" for key, value in row.items()) for row in rows[:limit]
    )


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
        source_required_sinks: dict[str, SourceRequiredSink] | None = None,
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
        self.source_required_sinks: dict[str, SourceRequiredSink] = source_required_sinks or {}
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

        # Optional semantic-judge layer: load FLOCK on this connection so enforcement can ask an LLM judge
        # (via llm_filter) for a second opinion when the deterministic hard gate blocks an action. Off unless
        # DFC_SEMANTIC_JUDGE is set; never touches generation.
        if semantic_judge.enabled():
            try:
                semantic_judge.setup(raw_conn)
                event_log.log("semantic_judge_enabled")
            except Exception as exc:
                event_log.log("semantic_judge_setup_failed", error=repr(exc)[:200])

        event_log.log("extraction_start", model=dfc_model)
        extracted_facts = extract_preamble_facts(
            llm,
            model=dfc_model,
            preamble=task_context.preamble,
            event_log=event_log,
        )
        # Always-present DENY SENTINEL: a value no real recipient/url/account can equal. Grounding an outbound
        # sink's destination on this (`recipient = PreambleData.no_authorized_destination`) blocks EVERY call to
        # that sink — the fail-safe for an effectful outbound channel the trusted task authorizes no destination
        # for (e.g. send_money on a buy-a-product task). Lets the generator default-deny unused egress sinks so
        # an injection can't introduce a brand-new exfil channel. See EXFILTRATION_POLICY_GUIDANCE.
        extracted_facts.setdefault("no_authorized_destination", "__DFC_NO_AUTHORIZED_DESTINATION__")
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
        source_required_sinks = build_source_required_index(
            generated_policies=generated_policies,
            registered_policy_ids=registered_policy_ids,
            runtime_schema=runtime_schema,
        )
        event_log.log(
            "policy_generation_complete",
            generated_count=len(generated_policies),
            registered_policy_ids=registered_policy_ids,
            deleted_count=len(deleted),
            total_repair_attempts=sum(record.repair_attempts for record in policy_registration),
            source_required_sinks=sorted(source_required_sinks),
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
            task_prompt=task_context.preamble,
            source_required_sinks=source_required_sinks,
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
            source_required_sinks=source_required_sinks,
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

    # --- SOURCE REQUIRED SQL gateway --------------------------------------------------------------
    def execute_source_required_sql(self, sql: str) -> SourceRequiredResult:
        """Run a model-authored SOURCE REQUIRED INSERT through the policy-aware connection. Returns a
        structured result; does NOT run the real tool (the adapter does so on status == 'ok'). On any
        block (validation/SQL error, policy KILL, 0-row filter, or judge-flagged injection) the caller
        surfaces the shared SR retry message and lets the agent retry."""
        try:
            sink = validate_source_required_sql(sql, self.source_required_sinks)
        except ValueError as exc:
            return SourceRequiredResult(status="error", sink_relation="", message=str(exc))

        relation = sink.sink_relation
        before = self._max_rowid(relation)
        try:
            self.conn.refresh_catalog()  # source *Output rows are written via raw_conn; refresh so the policy sees them
        except Exception:
            pass
        try:
            self.conn.execute(sql)
        except Exception as exc:
            msg = repr(exc).lower()  # engine KILL surfaces as "...executing the UDF..."
            status = "policy" if ("udf" in msg or "passant" in msg or "kill" in msg) else "error"
            return SourceRequiredResult(
                status=status, sink_relation=relation, tool_name=sink.tool_name, message=_short_error(exc)
            )

        rows = self._read_new_sink_rows(relation, list(sink.sink_columns), before)
        if not rows:
            return SourceRequiredResult(
                status="filtered", sink_relation=relation, tool_name=sink.tool_name,
                message="The data-flow policy filtered out the value (it was not grounded in the source).",
            )
        if semantic_judge.enabled() and self.run_source_required_judge(sink, rows):
            self._delete_new_sink_rows(relation, before)  # roll back the flagged write
            return SourceRequiredResult(
                status="injection", sink_relation=relation, tool_name=sink.tool_name,
                message="A value written appears to originate from a prompt injection in the source content.",
            )
        return SourceRequiredResult(status="ok", sink_relation=relation, tool_name=sink.tool_name, rows=rows)

    def run_source_required_judge(self, sink: SourceRequiredSink, sink_rows: list[dict[str, Any]]) -> bool:
        """Ask the content-provenance judge whether any sink value derives from a prompt injection in the
        source content. True = injection -> block. Logged to the validator's judge_decisions."""
        sink_values = _format_rows_for_judge(sink_rows)
        source_blocks = []
        for source in sink.source_relations:
            src_rows = self._read_source_rows(source, list(sink.source_columns.get(source) or {}))
            source_blocks.append(f"{source}:\n{_format_rows_for_judge(src_rows)}")
        flagged = semantic_judge.judge_source_injection(sink_values, "\n\n".join(source_blocks))
        self.validator.judge_decisions.append({
            "event_type": f"source_required:{sink.tool_name}",
            "sink": sink.sink_relation,
            "sources": sink.source_relations,
            "decision": "block" if flagged else "allow",
            "judge_model": getattr(semantic_judge, "_JUDGE_MODEL", "?").split("/")[-1],
            "judge_mode": "source_injection",
        })
        self.event_log.log(
            "source_required_judge", sink=sink.sink_relation, sources=sink.source_relations,
            decision="block" if flagged else "allow",
        )
        return flagged

    def _max_rowid(self, relation: str) -> int:
        try:
            row = self.raw_conn.execute(
                f"SELECT COALESCE(max(rowid), -1) FROM {quote_identifier(relation)}"
            ).fetchone()
            return int(row[0]) if row is not None else -1
        except Exception:
            return -1

    def _read_new_sink_rows(self, relation: str, columns: list[str], after_rowid: int) -> list[dict[str, Any]]:
        if not columns:
            return []
        col_sql = ", ".join(quote_identifier(column) for column in columns)
        rows = self.raw_conn.execute(
            f"SELECT {col_sql} FROM {quote_identifier(relation)} WHERE rowid > ? ORDER BY rowid",
            [after_rowid],
        ).fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def _delete_new_sink_rows(self, relation: str, after_rowid: int) -> None:
        try:
            self.raw_conn.execute(
                f"DELETE FROM {quote_identifier(relation)} WHERE rowid > ?", [after_rowid]
            )
        except Exception:
            pass

    def _read_source_rows(self, relation: str, columns: list[str]) -> list[dict[str, Any]]:
        try:
            if columns:
                col_sql = ", ".join(quote_identifier(column) for column in columns)
                rows = self.raw_conn.execute(f"SELECT {col_sql} FROM {quote_identifier(relation)}").fetchall()
                return [dict(zip(columns, row)) for row in rows]
            rows = self.raw_conn.execute(f"SELECT * FROM {quote_identifier(relation)}").fetchall()
            names = [desc[0] for desc in self.raw_conn.description]
            return [dict(zip(names, row)) for row in rows]
        except Exception:
            return []

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
