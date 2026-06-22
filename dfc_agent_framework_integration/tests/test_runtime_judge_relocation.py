from __future__ import annotations

import duckdb
from data_flow_control import Policy, dfc

from dfc_agent_framework_integration import semantic_judge
from dfc_agent_framework_integration.events import ensure_relation_write_tables
from dfc_agent_framework_integration.runtime import DFCRuntimeValidator
from dfc_agent_framework_integration.schema import GeneratedPolicy, RelationSchema, RuntimeSchema

SINK_COLS = {"recipient": "VARCHAR", "amount": "VARCHAR"}
DIM_PGN = (
    "SINK SendMoneyInput\n"
    "DIMENSION PreambleData\n"
    "CONSTRAINT SendMoneyInput.recipient = PreambleData.recipient\n"
    "ON FAIL KILL\n"
    "DESCRIPTION recipient must be the task-authorized account"
)


def _validator() -> DFCRuntimeValidator:
    raw = duckdb.connect()
    ensure_relation_write_tables(raw, "SendMoneyInput", SINK_COLS)
    raw.execute('CREATE TABLE "PreambleData" ("recipient" VARCHAR)')
    raw.execute("INSERT INTO PreambleData VALUES ('DE123')")
    conn = dfc(raw)
    conn.register_policy(Policy.from_pgn(DIM_PGN))
    conn.refresh_catalog()
    schema = RuntimeSchema(
        tool_input_relations=[RelationSchema(name="SendMoneyInput", tool_name="send_money", columns=SINK_COLS)],
    )
    policy = GeneratedPolicy(policy_id="d1", pgn=DIM_PGN, description="g",
                             applies_to_relation="SendMoneyInput", rationale="r")
    return DFCRuntimeValidator(raw, conn, schema, [policy], ["d1"], {}, task_id="t",
                              task_prompt="send money to alice DE123")


def test_validate_event_never_consults_judge_even_when_enabled(monkeypatch):
    def tripwire(*_a, **_k):
        raise AssertionError("the semantic judge must not run during validate_event")

    monkeypatch.setattr(semantic_judge, "enabled", lambda: True)
    monkeypatch.setattr(semantic_judge, "quarantine_enabled", lambda: True)
    monkeypatch.setattr(semantic_judge, "mode", lambda: "injection_detect")
    monkeypatch.setattr(semantic_judge, "judge_allows", tripwire)
    monkeypatch.setattr(semantic_judge, "any_injection", tripwire)

    validator = _validator()
    # Ungrounded value -> hard block; must return a violation WITHOUT consulting the judge.
    violation = validator.validate_tool_call("send_money", {"recipient": "ATTACKER", "amount": "1"})
    assert violation is not None
    assert violation.policy_ids == ["d1"]
    # Grounded value still passes.
    assert validator.validate_tool_call("send_money", {"recipient": "DE123", "amount": "1"}) is None
