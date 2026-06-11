from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENTS_FILENAME = "events.jsonl"
STATUS_FILENAME = "status.json"


def resolve_dfc_artifact_dir() -> Path | None:
    try:
        from agentdojo.logging import Logger, TraceLogger
    except ImportError:
        return None

    logger = Logger.get()
    if not isinstance(logger, TraceLogger):
        return None
    return logger.dfc_artifact_dir()


class DFCEventLog:
    """Append-only DFC progress log for live monitoring under the task *_dfc directory."""

    def __init__(self, output_dir: Path | None) -> None:
        self._output_dir = output_dir
        self._lock = threading.Lock()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._output_dir is not None

    @property
    def events_path(self) -> Path | None:
        if self._output_dir is None:
            return None
        return self._output_dir / EVENTS_FILENAME

    @property
    def status_path(self) -> Path | None:
        if self._output_dir is None:
            return None
        return self._output_dir / STATUS_FILENAME

    def log(self, event: str, **details: Any) -> None:
        if self._output_dir is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            events_path = self.events_path
            assert events_path is not None
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._write_status(record)

    def _write_status(self, latest_record: dict[str, Any]) -> None:
        status_path = self.status_path
        if status_path is None:
            return
        status = {
            "latest_event": latest_record["event"],
            "latest_timestamp": latest_record["timestamp"],
            "details": {
                key: value
                for key, value in latest_record.items()
                if key not in {"timestamp", "event"}
            },
        }
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
