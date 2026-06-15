from __future__ import annotations

import duckdb
from data_flow_control import Policy, dfc


EXAMPLE_9_PGN = """SINK SendEmailInput
DIMENSION (SELECT COALESCE(max(password), '') AS password FROM GetShoppingAccountUserInformationOutput) AccountInfo
CONSTRAINT AccountInfo.password = '' OR NOT contains(SendEmailInput.body, AccountInfo.password)
ON FAIL KILL
DESCRIPTION Prevent data exfiltration: do not email the account password retrieved during this run."""


def test_example_9_registers_and_allows_email_before_account_read():
    raw = duckdb.connect()
    raw.execute("CREATE TABLE GetShoppingAccountUserInformationOutput (password VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInput (body VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInputWriteStaging (body VARCHAR, __dfc_event_id VARCHAR)")

    conn = dfc(raw)
    conn.register_policy(Policy.from_pgn(EXAMPLE_9_PGN))
    conn.refresh_catalog()

    raw.execute("INSERT INTO SendEmailInputWriteStaging VALUES ('hello', 'e1')")
    conn.execute("INSERT INTO SendEmailInput SELECT body, __dfc_event_id FROM SendEmailInputWriteStaging")


def test_example_9_kills_when_body_contains_observed_password():
    raw = duckdb.connect()
    raw.execute("CREATE TABLE GetShoppingAccountUserInformationOutput (password VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInput (body VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInputWriteStaging (body VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("INSERT INTO GetShoppingAccountUserInformationOutput VALUES ('secret', 'e0')")

    conn = dfc(raw)
    conn.register_policy(Policy.from_pgn(EXAMPLE_9_PGN))
    conn.refresh_catalog()

    raw.execute("INSERT INTO SendEmailInputWriteStaging VALUES ('Your password is: secret', 'e1')")
    try:
        conn.execute("INSERT INTO SendEmailInput SELECT body, __dfc_event_id FROM SendEmailInputWriteStaging")
        raised = False
    except Exception:
        raised = True
    assert raised


def test_example_9_allows_email_when_password_not_in_body():
    raw = duckdb.connect()
    raw.execute("CREATE TABLE GetShoppingAccountUserInformationOutput (password VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInput (body VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("CREATE TABLE SendEmailInputWriteStaging (body VARCHAR, __dfc_event_id VARCHAR)")
    raw.execute("INSERT INTO GetShoppingAccountUserInformationOutput VALUES ('secret', 'e0')")

    conn = dfc(raw)
    conn.register_policy(Policy.from_pgn(EXAMPLE_9_PGN))
    conn.refresh_catalog()

    raw.execute("INSERT INTO SendEmailInputWriteStaging VALUES ('Order shipped, no credentials included.', 'e1')")
    conn.execute("INSERT INTO SendEmailInput SELECT body, __dfc_event_id FROM SendEmailInputWriteStaging")
