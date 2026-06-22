from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dfc_agent_framework_integration.events import quote_identifier

METADATA_FILENAME = "metadata.json"
DATABASE_FILENAME = "database.duckdb"


def table_inventory(raw_conn: Any) -> dict[str, int]:
    tables = raw_conn.execute("SHOW TABLES").fetchall()
    inventory: dict[str, int] = {}
    for (table_name,) in tables:
        quoted = quote_identifier(table_name)
        count = raw_conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
        inventory[table_name] = int(count)
    return inventory


def serialize_registered_policies(dfc_conn: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    try:
        policies = dfc_conn.policies()
    except Exception:
        return serialized

    for policy in policies:
        serialized.append(
            {
                "constraint": policy.constraint,
                "on_fail": policy.on_fail_label,
                "sources": list(policy.sources),
                "required_sources": list(policy.required_sources or []),
                "sink": policy.sink,
                "dimensions": list(policy.dimensions or []),
                "description": policy.description,
            }
        )
    return serialized


def policy_fire_counts_for_export(context: Any) -> dict[str, int]:
    counts = dict(context.diagnostics.policy_fire_counts)
    for policy_id in context.registered_policy_ids:
        counts.setdefault(policy_id, 0)
    return counts


def policy_generation_summary(context: Any) -> dict[str, Any]:
    records = context.diagnostics.policy_registration
    return {
        "total_repair_attempts": sum(record.repair_attempts for record in records),
        "policies": [record.model_dump() for record in records],
    }


def build_metadata_document(context: Any) -> dict[str, Any]:
    runtime_schema: RuntimeSchema = context.runtime_schema
    return {
        "task_context": context.task_context.model_dump(),
        "dfc_model": context.dfc_model,
        "agent_model": context.agent_model,
        "model": context.dfc_model,
        "runtime_schema": runtime_schema.model_dump(),
        "extracted_facts": context.extracted_facts,
        "generated_policies": [policy.model_dump() for policy in context.generated_policies],
        "registered_policy_ids": list(context.registered_policy_ids),
        "registered_passant_policies": serialize_registered_policies(context.conn),
        "deleted_policies": [record.model_dump() for record in context.diagnostics.deleted_policies],
        "policy_generation": policy_generation_summary(context),
        "policy_fire_counts": policy_fire_counts_for_export(context),
        "validation_events": list(context.diagnostics.validation_events),
        "judge_decisions": list(getattr(context.validator, "judge_decisions", [])),
        "table_inventory": table_inventory(context.raw_conn),
    }


def export_duckdb_database(raw_conn: Any, database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    escaped_path = str(database_path.resolve()).replace("'", "''")
    raw_conn.execute(f"ATTACH '{escaped_path}' AS dfc_export_target")
    try:
        raw_conn.execute("COPY FROM DATABASE memory TO dfc_export_target")
    finally:
        raw_conn.execute("DETACH dfc_export_target")


def export_run_artifacts(context: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILENAME
    database_path = output_dir / DATABASE_FILENAME

    metadata = build_metadata_document(context)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_duckdb_database(context.raw_conn, database_path)
    event_log = getattr(context, "event_log", None)
    if event_log is not None and event_log.enabled:
        event_log.log("artifacts_exported", metadata_path=str(metadata_path), database_path=str(database_path))


def append_policy_ledger(context: Any, ledger_path: Path) -> None:
    """Append one record per task to a central, append-only JSONL — every policy generated for the run, in
    one greppable place (the per-task metadata.json files remain the full record; this is the index across
    them). Best-effort: never let ledger I/O break a run."""
    import time

    from pydantic import BaseModel

    task_ctx = getattr(context, "task_context", None)
    registered = set(getattr(context, "registered_policy_ids", []) or [])
    diagnostics = getattr(context, "diagnostics", None)
    deleted = getattr(diagnostics, "deleted_policies", []) if diagnostics is not None else []
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": getattr(task_ctx, "benchmark_name", None),
        "suite": getattr(task_ctx, "suite_name", None),
        "task_id": getattr(task_ctx, "task_id", None),
        "dfc_model": getattr(context, "dfc_model", None),
        "agent_model": getattr(context, "agent_model", None),
        "extracted_facts": getattr(context, "extracted_facts", {}),
        "registered_policy_ids": sorted(registered),
        "policies": [
            {
                "policy_id": p.policy_id,
                "sink": p.applies_to_relation,
                "description": p.description,
                "pgn": p.pgn,
                "registered": p.policy_id in registered,
            }
            for p in getattr(context, "generated_policies", [])
        ],
        "deleted": [{"policy_id": d.policy_id, "rationale": d.rationale} for d in deleted],
    }
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o)) + "\n")
