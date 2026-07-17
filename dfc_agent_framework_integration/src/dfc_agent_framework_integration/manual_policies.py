from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from data_flow_control import Policy

from dfc_agent_framework_integration.dfc_event_log import DFCEventLog
from dfc_agent_framework_integration.repair import validate_policy_catalog
from dfc_agent_framework_integration.schema import (
    DeletedPolicyRecord,
    GeneratedPolicy,
    PolicyRegistrationRecord,
    RuntimeSchema,
)

_POLICY_START_RE = re.compile(r"(?=^\s*SINK\s+\w+)", re.IGNORECASE | re.MULTILINE)
_SINK_RE = re.compile(r"^\s*SINK\s+(\w+)", re.IGNORECASE | re.MULTILINE)
_DESCRIPTION_RE = re.compile(r"^\s*DESCRIPTION\s+(.+)$", re.IGNORECASE | re.MULTILINE)


def _split_policy_bundle(text: str) -> list[str]:
    return [part.strip() for part in _POLICY_START_RE.split(text) if part.strip()]


def _manual_policy_record(*, path: Path, index: int, pgn: str, parsed: Policy | None = None) -> GeneratedPolicy:
    sink_match = _SINK_RE.search(pgn)
    sink = sink_match.group(1) if sink_match else ""
    description_match = _DESCRIPTION_RE.search(pgn)
    description = description_match.group(1).strip() if description_match else parsed.description if parsed else ""
    return GeneratedPolicy(
        policy_id=f"manual_{path.stem}_{index}",
        pgn=pgn,
        description=description,
        applies_to_relation=sink,
        applies_to_event=None,
        rationale=f"Handwritten policy loaded from {path.as_posix()}.",
    )


def load_and_register_manual_policies(
    conn: Any,
    *,
    policy_paths: list[str | Path],
    runtime_schema: RuntimeSchema,
    event_log: DFCEventLog | None = None,
) -> tuple[list[GeneratedPolicy], list[str], list[DeletedPolicyRecord], list[PolicyRegistrationRecord]]:
    event_log = event_log or DFCEventLog(None)
    generated: list[GeneratedPolicy] = []
    registered_ids: list[str] = []
    deleted: list[DeletedPolicyRecord] = []
    registration_records: list[PolicyRegistrationRecord] = []

    for raw_path in policy_paths:
        path = Path(raw_path)
        event_log.log("manual_policy_bundle_load_start", path=str(path))
        try:
            blocks = _split_policy_bundle(path.read_text(encoding="utf-8"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            event_log.log("manual_policy_bundle_load_failed", path=str(path), error=error)
            deleted.append(
                DeletedPolicyRecord(
                    policy_id=f"manual_{path.stem}",
                    rationale="Manual policy bundle could not be read.",
                    error=error,
                )
            )
            continue

        for index, pgn in enumerate(blocks, start=1):
            record: GeneratedPolicy | None = None
            try:
                parsed = Policy.from_pgn(pgn)
                record = _manual_policy_record(path=path, index=index, pgn=pgn, parsed=parsed)
                event_log.log("manual_policy_register_start", policy_id=record.policy_id, path=str(path))
                validate_policy_catalog(parsed, runtime_schema)
                conn.register_policy(parsed)
                generated.append(record)
                registered_ids.append(record.policy_id)
                registration_records.append(
                    PolicyRegistrationRecord(
                        policy_id=record.policy_id,
                        repair_attempts=0,
                        outcome="registered",
                    )
                )
                event_log.log("manual_policy_register_success", policy_id=record.policy_id)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                fallback = record or _manual_policy_record(path=path, index=index, pgn=pgn)
                generated.append(fallback)
                deleted.append(
                    DeletedPolicyRecord(
                        policy_id=fallback.policy_id,
                        rationale="Manual policy did not parse or did not match the runtime schema.",
                        error=error,
                    )
                )
                registration_records.append(
                    PolicyRegistrationRecord(
                        policy_id=fallback.policy_id,
                        repair_attempts=0,
                        outcome="deleted",
                    )
                )
                event_log.log("manual_policy_register_deleted", policy_id=fallback.policy_id, error=error)

        event_log.log("manual_policy_bundle_load_complete", path=str(path), policy_count=len(blocks))

    return generated, registered_ids, deleted, registration_records
