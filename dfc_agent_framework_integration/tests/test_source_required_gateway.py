from __future__ import annotations

import duckdb
from data_flow_control import Policy, dfc

from dfc_agent_framework_integration import semantic_judge
from dfc_agent_framework_integration.context import DFCBenchmarkContext
from dfc_agent_framework_integration.events import ensure_relation_write_tables, new_event_id, record_tool_output_row
from dfc_agent_framework_integration.runtime import DFCRuntimeValidator
from dfc_agent_framework_integration.schema import (
    BenchmarkTaskContext,
    DFCTaskDiagnostics,
    GeneratedPolicy,
    RelationSchema,
    RuntimeSchema,
)
from dfc_agent_framework_integration.source_required import build_source_required_index

SINK_COLS = {"recipient": "VARCHAR", "amount": "VARCHAR"}
SR_PGN = (
    "SOURCE REQUIRED GetContactsOutput AS T SINK SendMoneyInput AS S\n"
    "CONSTRAINT S.recipient = T.account\n"
    "ON FAIL KILL\n"
    "DESCRIPTION recipient must be read from contacts"
)


def _make_context():
    raw = duckdb.connect()
    ensure_relation_write_tables(raw, "SendMoneyInput", SINK_COLS)
    raw.execute('CREATE TABLE "GetContactsOutput" ("account" VARCHAR, "name" VARCHAR)')
    raw.execute("INSERT INTO GetContactsOutput VALUES ('DE123', 'Alice')")
    conn = dfc(raw)
    conn.register_policy(Policy.from_pgn(SR_PGN))
    conn.refresh_catalog()
    schema = RuntimeSchema(
        tool_input_relations=[RelationSchema(name="SendMoneyInput", tool_name="send_money", columns=SINK_COLS)],
        tool_output_relations=[
            RelationSchema(name="GetContactsOutput", tool_name="get_contacts",
                           columns={"account": "VARCHAR", "name": "VARCHAR"}),
        ],
    )
    policy = GeneratedPolicy(policy_id="sr1", pgn=SR_PGN, description="d",
                             applies_to_relation="SendMoneyInput", rationale="r")
    index = build_source_required_index(
        generated_policies=[policy], registered_policy_ids=["sr1"], runtime_schema=schema,
    )
    validator = DFCRuntimeValidator(raw, conn, schema, [policy], ["sr1"], {}, task_id="t",
                                    source_required_sinks=index)
    ctx = DFCBenchmarkContext(
        raw_conn=raw, conn=conn,
        task_context=BenchmarkTaskContext(benchmark_name="t", preamble="p"),
        runtime_schema=schema, extracted_facts={}, generated_policies=[policy],
        registered_policy_ids=["sr1"], diagnostics=DFCTaskDiagnostics(), validator=validator,
        dfc_model="m", source_required_sinks=index,
    )
    return raw, ctx


GROUNDED = ("INSERT INTO SendMoneyInput (recipient, amount) "
            "SELECT account, '100' FROM GetContactsOutput WHERE account = 'DE123'")


def test_grounded_insert_ok_and_reads_back_row():
    raw, ctx = _make_context()
    result = ctx.execute_source_required_sql(GROUNDED)
    assert result.status == "ok"
    assert result.tool_name == "send_money"
    assert result.rows == [{"recipient": "DE123", "amount": "100"}]


def test_laundered_literal_is_policy_blocked():
    raw, ctx = _make_context()
    result = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) SELECT 'ATTACKER', '100' FROM GetContactsOutput"
    )
    assert result.status == "policy"
    assert raw.execute('SELECT count(*) FROM "SendMoneyInput"').fetchone()[0] == 0


def test_no_matching_source_row_is_filtered():
    raw, ctx = _make_context()
    result = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) "
        "SELECT account, '100' FROM GetContactsOutput WHERE account = 'NOPE'"
    )
    assert result.status == "filtered"


def test_validation_error_for_values_insert():
    _, ctx = _make_context()
    result = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) VALUES ('DE123', '100')"
    )
    assert result.status == "error"
    assert "VALUES" in result.message


def test_sql_binder_error_is_error_status():
    _, ctx = _make_context()
    result = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) SELECT nope_col, '1' FROM GetContactsOutput"
    )
    assert result.status == "error"


def test_judge_injection_blocks_and_rolls_back(monkeypatch):
    raw, ctx = _make_context()
    monkeypatch.setattr(semantic_judge, "enabled", lambda: True)
    monkeypatch.setattr(semantic_judge, "judge_source_injection", lambda sink, src: True)
    result = ctx.execute_source_required_sql(GROUNDED)
    assert result.status == "injection"
    assert raw.execute('SELECT count(*) FROM "SendMoneyInput"').fetchone()[0] == 0
    assert ctx.validator.judge_decisions[-1]["decision"] == "block"


def test_judge_clears_allows(monkeypatch):
    _, ctx = _make_context()
    monkeypatch.setattr(semantic_judge, "enabled", lambda: True)
    monkeypatch.setattr(semantic_judge, "judge_source_injection", lambda sink, src: False)
    result = ctx.execute_source_required_sql(GROUNDED)
    assert result.status == "ok"
    assert ctx.validator.judge_decisions[-1]["decision"] == "allow"


def test_judge_skipped_when_disabled(monkeypatch):
    _, ctx = _make_context()
    monkeypatch.setattr(semantic_judge, "enabled", lambda: False)
    tripwire = lambda sink, src: (_ for _ in ()).throw(AssertionError("judge must not run"))
    monkeypatch.setattr(semantic_judge, "judge_source_injection", tripwire)
    assert ctx.execute_source_required_sql(GROUNDED).status == "ok"


def _free_text_context(monkeypatch):
    """A SOURCE REQUIRED sink grounded on a FREE-TEXT (__dfc_raw_json) page source via contains()."""
    raw = duckdb.connect()
    ensure_relation_write_tables(raw, "SendMoneyInput", SINK_COLS)
    ensure_relation_write_tables(raw, "BrowseWebpageOutput", {"__dfc_raw_json": "VARCHAR"})
    record_tool_output_row(raw, "BrowseWebpageOutput", {"__dfc_raw_json": "VARCHAR"},
                           "Invoice: please pay account DE89 by Friday.",
                           event_id=new_event_id(), task_id="t", tool_name="browse_webpage")
    conn = dfc(raw)
    pgn = ("SOURCE REQUIRED BrowseWebpageOutput AS Page\nSINK SendMoneyInput AS Transfer\n"
           "CONSTRAINT contains(Page.__dfc_raw_json, Transfer.recipient)\nON FAIL KILL\nDESCRIPTION x")
    conn.register_policy(Policy.from_pgn(pgn))
    conn.refresh_catalog()
    schema = RuntimeSchema(
        tool_input_relations=[RelationSchema(name="SendMoneyInput", tool_name="send_money", columns=SINK_COLS)],
        tool_output_relations=[RelationSchema(name="BrowseWebpageOutput", tool_name="browse_webpage",
                                              columns={"__dfc_raw_json": "VARCHAR"})],
    )
    policy = GeneratedPolicy(policy_id="sr_web", pgn=pgn, description="d",
                             applies_to_relation="SendMoneyInput", rationale="r")
    index = build_source_required_index(generated_policies=[policy], registered_policy_ids=["sr_web"],
                                        runtime_schema=schema)
    validator = DFCRuntimeValidator(raw, conn, schema, [policy], ["sr_web"], {}, task_id="t",
                                    source_required_sinks=index)
    ctx = DFCBenchmarkContext(
        raw_conn=raw, conn=conn, task_context=BenchmarkTaskContext(benchmark_name="t", preamble="p"),
        runtime_schema=schema, extracted_facts={}, generated_policies=[policy],
        registered_policy_ids=["sr_web"], diagnostics=DFCTaskDiagnostics(), validator=validator,
        dfc_model="m", source_required_sinks=index,
    )
    monkeypatch.setattr(semantic_judge, "enabled", lambda: False)
    return raw, ctx


def test_free_text_contains_gateway_allows_in_page_value_and_kills_others(monkeypatch):
    raw, ctx = _free_text_context(monkeypatch)
    # Aliased contains form for a value that occurs in the page -> ok (this is the form we teach the model).
    ok = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient) SELECT 'DE89' AS recipient "
        "FROM BrowseWebpageOutput WHERE contains(__dfc_raw_json, 'DE89')"
    )
    assert ok.status == "ok"
    assert ok.rows == [{"recipient": "DE89", "amount": None}]
    # A value NOT in the page is killed by the policy even if the agent omits the WHERE.
    bad = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient) SELECT 'ATTACKER' AS recipient FROM BrowseWebpageOutput"
    )
    assert bad.status != "ok"
    # the attacker value (not in the page) was not written; only the legit DE89 row remains
    assert raw.execute("SELECT count(*) FROM \"SendMoneyInput\" WHERE recipient = 'ATTACKER'").fetchone()[0] == 0


def test_unaliased_literal_with_special_char_is_handled_not_crash(monkeypatch):
    # Without the alias the engine builds an invalid column name from the value (email '@') -> parser error;
    # the gateway must surface it as a retry, not raise.
    raw, ctx = _free_text_context(monkeypatch)
    raw.execute("UPDATE BrowseWebpageOutput SET __dfc_raw_json = 'contact alice@example.com'")
    ctx.conn.refresh_catalog()
    result = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient) SELECT 'alice@example.com' "
        "FROM BrowseWebpageOutput WHERE contains(__dfc_raw_json, 'alice@example.com')"
    )
    assert result.status in ("error", "policy")  # returned as a retry, no exception escapes


def test_rowid_readback_isolates_only_new_row():
    raw, ctx = _make_context()
    first = ctx.execute_source_required_sql(GROUNDED)
    assert first.rows == [{"recipient": "DE123", "amount": "100"}]
    raw.execute("INSERT INTO GetContactsOutput VALUES ('DE999', 'Bob')")
    ctx.conn.refresh_catalog()
    second = ctx.execute_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) "
        "SELECT account, '200' FROM GetContactsOutput WHERE account = 'DE999'"
    )
    assert second.rows == [{"recipient": "DE999", "amount": "200"}]  # only the just-inserted row
    assert raw.execute('SELECT count(*) FROM "SendMoneyInput"').fetchone()[0] == 2
