from __future__ import annotations

from typing import Any

from dfc_agent_framework_integration.events import PREAMBLE_RELATION, quote_identifier
from dfc_agent_framework_integration.extraction import validate_extraction_facts


def materialize_preamble_data(raw_conn: Any, facts: dict[str, str], relation_name: str = PREAMBLE_RELATION) -> None:
    validate_extraction_facts(facts)
    quoted_relation = quote_identifier(relation_name)
    raw_conn.execute(f"DROP TABLE IF EXISTS {quoted_relation}")
    if not facts:
        raise ValueError("Cannot materialize empty preamble facts")
    col_defs = ", ".join(f"{quote_identifier(key)} VARCHAR" for key in facts)
    raw_conn.execute(f"CREATE TABLE {quoted_relation} ({col_defs})")
    columns = list(facts.keys())
    placeholders = ", ".join("?" for _ in columns)
    quoted_cols = ", ".join(quote_identifier(column) for column in columns)
    values = [facts[column] for column in columns]
    raw_conn.execute(
        f"INSERT INTO {quoted_relation} ({quoted_cols}) VALUES ({placeholders})",
        values,
    )


def fetch_preamble_row(raw_conn: Any, relation_name: str = PREAMBLE_RELATION) -> dict[str, str]:
    quoted_relation = quote_identifier(relation_name)
    rows = raw_conn.execute(f"SELECT * FROM {quoted_relation}").fetchall()
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {relation_name}, found {len(rows)}")
    columns = [column[0] for column in raw_conn.description]
    return {column: "" if value is None else str(value) for column, value in zip(columns, rows[0])}
