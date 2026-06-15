from __future__ import annotations

import duckdb
import pytest

from dfc_agent_framework_integration.materialize import fetch_preamble_row, materialize_preamble_data


def test_materialize_preamble_data_creates_one_row():
    raw = duckdb.connect()
    facts = {
        "authorized_recipient_email": "alice@example.com",
        "invoice_save_dir": "/user/invoices",
    }
    materialize_preamble_data(raw, facts)
    row = fetch_preamble_row(raw)
    assert row == facts


def test_invalid_column_names_rejected_before_ddl():
    raw = duckdb.connect()
    with pytest.raises(Exception):
        materialize_preamble_data(raw, {"Bad-Key": "value"})
