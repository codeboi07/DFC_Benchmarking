from __future__ import annotations

import json
from pathlib import Path

import duckdb

from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.persistence import (
    DATABASE_FILENAME,
    METADATA_FILENAME,
    export_run_artifacts,
)
from dfc_agent_framework_integration.schema import BenchmarkTaskContext, RuntimeSchema


def test_export_run_artifacts_writes_metadata_and_database(tmp_path, combined_llm, send_email_runtime):
    context = DFCBenchmarkContext.prepare_task(
        task_context=BenchmarkTaskContext(
            benchmark_name="agentdyn",
            suite_name="shopping",
            task_id="user_task_0",
            preamble="Email alice@example.com",
        ),
        runtime_schema=RuntimeSchema.from_tools(send_email_runtime.functions),
        llm=combined_llm,
        dfc_model="fake-model",
        functions=send_email_runtime.functions,
    )
    try:
        output_dir = tmp_path / "injection_task_0_dfc"
        export_run_artifacts(context, output_dir)

        metadata_path = output_dir / METADATA_FILENAME
        database_path = output_dir / DATABASE_FILENAME
        assert metadata_path.exists()
        assert database_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["extracted_facts"]["authorized_recipient_email"] == "alice@example.com"
        assert metadata["generated_policies"][0]["policy_id"] == "send_email_recipient"
        assert "PreambleData" in metadata["table_inventory"]
        assert metadata["registered_policy_ids"] == ["send_email_recipient"]
        assert metadata["policy_generation"]["total_repair_attempts"] == 0
        assert metadata["policy_generation"]["policies"] == [
            {
                "policy_id": "send_email_recipient",
                "repair_attempts": 0,
                "outcome": "registered",
            }
        ]
        assert metadata["policy_fire_counts"] == {"send_email_recipient": 0}
        send_email_input = metadata["runtime_schema"]["tool_input_relations"][0]
        send_email_output = metadata["runtime_schema"]["tool_output_relations"][0]
        assert send_email_input["description"]
        assert send_email_output["description"]
        assert send_email_input["tool_name"] == "send_email"
        assert send_email_output["tool_name"] == "send_email"

        exported = duckdb.connect(str(database_path))
        try:
            tables = {name for (name,) in exported.execute("SHOW TABLES").fetchall()}
            assert "PreambleData" in tables
            row = exported.execute(
                "SELECT authorized_recipient_email FROM PreambleData"
            ).fetchone()
            assert row == ("alice@example.com",)
        finally:
            exported.close()
    finally:
        context.close()
