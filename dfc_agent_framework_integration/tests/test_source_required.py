from __future__ import annotations

import pytest

from dfc_agent_framework_integration.schema import (
    GeneratedPolicy,
    RelationSchema,
    RuntimeSchema,
)
from dfc_agent_framework_integration.source_required import (
    build_source_required_index,
    format_source_required_redirect,
    format_source_required_system_notice,
    source_required_sink_for_tool,
    validate_source_required_sql,
)

SR_PGN = (
    "SOURCE REQUIRED GetContactsOutput AS T SINK SendMoneyInput AS S\n"
    "CONSTRAINT S.recipient = T.account\n"
    "ON FAIL KILL\n"
    "DESCRIPTION The recipient must have been read from your contacts."
)
DIM_PGN = (
    "SINK SendMoneyInput\n"
    "DIMENSION PreambleData\n"
    "CONSTRAINT SendMoneyInput.recipient = PreambleData.recipient\n"
    "ON FAIL KILL\n"
    "DESCRIPTION grounded to preamble"
)


def _schema() -> RuntimeSchema:
    return RuntimeSchema(
        tool_input_relations=[
            RelationSchema(
                name="SendMoneyInput", tool_name="send_money",
                columns={"recipient": "VARCHAR", "amount": "VARCHAR"},
                column_descriptions={"recipient": "the payee account"},
            ),
            RelationSchema(name="SearchInput", tool_name="search", columns={"query": "VARCHAR"}),
        ],
        tool_output_relations=[
            RelationSchema(
                name="GetContactsOutput", tool_name="get_contacts",
                columns={"account": "VARCHAR", "name": "VARCHAR"},
                column_descriptions={"account": "the contact's account number"},
            ),
        ],
    )


def _policy(policy_id: str, pgn: str, sink: str = "SendMoneyInput") -> GeneratedPolicy:
    return GeneratedPolicy(
        policy_id=policy_id, pgn=pgn, description="d", applies_to_relation=sink, rationale="r",
    )


def _index(policies=None, registered=None):
    policies = policies if policies is not None else [_policy("sr1", SR_PGN)]
    registered = registered if registered is not None else [p.policy_id for p in policies]
    return build_source_required_index(
        generated_policies=policies, registered_policy_ids=registered, runtime_schema=_schema(),
    )


def test_index_captures_source_required_sink():
    index = _index()
    assert set(index) == {"SendMoneyInput"}
    sink = index["SendMoneyInput"]
    assert sink.tool_name == "send_money"
    assert sink.source_relations == ["GetContactsOutput"]
    assert sink.sink_columns == {"recipient": "VARCHAR", "amount": "VARCHAR"}
    assert sink.source_columns["GetContactsOutput"] == {"account": "VARCHAR", "name": "VARCHAR"}
    assert source_required_sink_for_tool(index, "send_money") is sink
    assert source_required_sink_for_tool(index, "search") is None


def test_index_ignores_dimension_and_unregistered_policies():
    assert _index([_policy("dim1", DIM_PGN)]) == {}
    # SR policy present but not registered -> excluded
    assert _index([_policy("sr1", SR_PGN)], registered=[]) == {}


def test_validate_accepts_grounded_select_from_source():
    index = _index()
    sink = validate_source_required_sql(
        "INSERT INTO SendMoneyInput (recipient, amount) "
        "SELECT account, '100' FROM GetContactsOutput WHERE account = 'DE123'",
        index,
    )
    assert sink.sink_relation == "SendMoneyInput"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO SendMoneyInput (recipient, amount) VALUES ('x', '1')",  # VALUES bypass
        "INSERT INTO SendMoneyInput (recipient, amount) SELECT q, '1' FROM PreambleData",  # not the source
        "INSERT INTO SearchInput (query) SELECT account FROM GetContactsOutput",  # non-SR sink
        "INSERT INTO SendMoneyInput (recipient) SELECT account FROM GetContactsOutput; DROP TABLE x",  # multi
        "INSERT INTO SendMoneyInput (recipient) SELECT account FROM GetContactsOutput UNION SELECT 1",  # union
        "CREATE TABLE evil (x INT)",  # ddl
        "   ",  # empty
    ],
)
def test_validate_rejects(sql):
    with pytest.raises(ValueError):
        validate_source_required_sql(sql, _index())


def test_redirect_text_has_schemas_and_example():
    text = format_source_required_redirect(_index()["SendMoneyInput"])
    for needle in ("execute_sql_for_source_required", "SendMoneyInput", "recipient",
                   "GetContactsOutput", "account", "INSERT INTO"):
        assert needle in text


def test_system_notice_lists_policies_and_is_empty_when_none():
    notice = format_source_required_system_notice(_index())
    assert "send_money" in notice and "GetContactsOutput" in notice
    assert format_source_required_system_notice({}) == ""


FREE_TEXT_PGN = (
    "SOURCE REQUIRED BrowseWebpageOutput AS Page\n"
    "SINK SendMoneyInput AS Transfer\n"
    "CONSTRAINT contains(Page.__dfc_raw_json, Transfer.recipient)\n"
    "ON FAIL KILL\n"
    "DESCRIPTION recipient must occur in the fetched bill"
)


def _free_text_index():
    schema = RuntimeSchema(
        tool_input_relations=[
            RelationSchema(name="SendMoneyInput", tool_name="send_money",
                           columns={"recipient": "VARCHAR", "amount": "VARCHAR"}),
        ],
        tool_output_relations=[
            RelationSchema(name="BrowseWebpageOutput", tool_name="browse_webpage",
                           columns={"__dfc_raw_json": "VARCHAR"}),
        ],
    )
    policy = _policy("sr_web", FREE_TEXT_PGN)
    return build_source_required_index(
        generated_policies=[policy], registered_policy_ids=["sr_web"], runtime_schema=schema,
    )


def test_example_insert_uses_contains_and_aliases_value_for_free_text_source():
    # The example we hand the model MUST use contains() against __dfc_raw_json and alias the value to its
    # sink column — otherwise the engine derives an invalid column name from the literal (e.g. an email @).
    text = format_source_required_redirect(_free_text_index()["SendMoneyInput"])
    assert "contains(BrowseWebpageOutput.__dfc_raw_json" in text
    assert "AS recipient" in text
    # The example INSERT itself must not use a VALUES clause (only the prose mentions it is rejected).
    example_line = next(line for line in text.splitlines() if "INSERT INTO" in line)
    assert "VALUES" not in example_line
