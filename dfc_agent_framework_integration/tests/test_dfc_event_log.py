from __future__ import annotations

import json
from pathlib import Path

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog, EVENTS_FILENAME, STATUS_FILENAME


def test_event_log_appends_jsonl_and_updates_status(tmp_path: Path):
    event_log = DFCEventLog(tmp_path)
    assert event_log.enabled

    event_log.log("prepare_task_start", task_id="user_task_0")
    event_log.log("extraction_start", model="gpt-5.2")

    events_path = tmp_path / EVENTS_FILENAME
    status_path = tmp_path / STATUS_FILENAME
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["event"] == "prepare_task_start"
    assert first["task_id"] == "user_task_0"
    assert "timestamp" in first

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["latest_event"] == "extraction_start"
    assert status["details"]["model"] == "gpt-5.2"


def test_disabled_event_log_is_noop():
    event_log = DFCEventLog(None)
    assert not event_log.enabled
    event_log.log("ignored")
