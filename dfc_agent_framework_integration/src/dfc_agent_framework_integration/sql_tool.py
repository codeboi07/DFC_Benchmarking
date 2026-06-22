"""The SOURCE REQUIRED SQL gateway tool.

This is a stub: the agent's call is intercepted by the DFC tools executor (which holds the per-task
context, the policy-aware connection, and the real tool dispatch) before it ever reaches this body.
It exists only to be advertised to the model with a schema. The adapter injects it into the
functions runtime only when the task actually has a SOURCE REQUIRED sink.
"""
from __future__ import annotations

SQL_TOOL_NAME = "execute_sql_for_source_required"


def execute_sql_for_source_required(sql: str) -> str:
    """Perform a SOURCE REQUIRED-protected action by writing its sink with a single SQL INSERT.

    Some actions are protected by a data-flow policy that requires their value to be read from a
    specific source table (for example, a payment recipient that must occur in a bill you fetched). For
    those actions you cannot call the tool directly; call this instead with ONE INSERT that reads from
    the source table, e.g. `INSERT INTO <SinkTable> (<column>) SELECT '<value>' AS <column> FROM
    <SourceTable> WHERE ...`. Always ALIAS the selected value to its sink column (`AS <column>`) or the
    SQL will not parse. For a free-text source (an email or page stored as `__dfc_raw_json`) require the
    value to occur in what you read: `WHERE contains(<SourceTable>.__dfc_raw_json, '<value>')`. The
    `FROM <SourceTable>` clause is REQUIRED — it is what proves the value came from the source (the value
    itself is written as a literal). A SELECT with no FROM, or a bare `INSERT ... VALUES`, is rejected.
    On success the underlying action is performed and its result returned.

    :param sql: a single INSERT statement that reads its values from the required source table.
    """
    return ""
