"""DFCSession — the per-task DFC resource: a DuckDB connection wrapped by Passant, the real-tool
runtime, and the SQL interpreter that turns a guarded INSERT intent into a real tool call.

Two enforcement modes:
  - VALUE sinks (send_money/send_email/password/download_url): Passant `transform_query` grounds the
    value against `trusted`; 0 rows = blocked. Deterministic provenance.
  - URL sinks (browse_webpage/input_to_webpage, only when gate_urls=True): the interpreter checks the
    URL's domain against the `trusted_urls` allowlist. Heuristic, not provenance (see spec)."""

import ast
import json
import re

import duckdb
from data_flow_control import dfc

from agentdojo.defenses.dfc.gateway import schema_doc
from agentdojo.defenses.dfc.materialize import (
    GUARDED_SINKS,
    URL_GUARDED_SINKS,
    _domain,
    create_relations,
)
from agentdojo.defenses.dfc.policies import get_policies


def _to_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    s = str(v).strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            r = parse(s)
            return r if isinstance(r, list) else [r]
        except Exception:
            pass
    return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


class DFCSession:
    def __init__(self, env, real_runtime, prompt: str, suite_name: str | None, gate_urls: bool = False):
        self.real_runtime = real_runtime
        self.suite_name = suite_name
        self.gate_urls = gate_urls
        self.raw = duckdb.connect()
        create_relations(self.raw, prompt, env, gate_urls=gate_urls)
        self.conn = dfc(self.raw)
        for policy in get_policies(suite_name):
            self.conn.register_policy(policy)
        self.conn.refresh_catalog()
        self.value_sinks = GUARDED_SINKS
        self.url_sinks = URL_GUARDED_SINKS if gate_urls else {}
        self.trusted_domains = (
            {r[0] for r in self.raw.execute("SELECT value FROM trusted_urls").fetchall()}
            if gate_urls else set()
        )

    def schema_doc(self) -> str:
        return schema_doc(self.raw, self.value_sinks, self.url_sinks)

    # ---- SQL interpreter -----------------------------------------------------------------
    def handle_intent(self, sql: str, env) -> tuple[str, str | None]:
        sql = sql.strip().rstrip(";")
        if ";" in sql:
            return "", "Only one SQL statement per call (no semicolons)."
        if re.search(r"\bunion\b", sql, re.I):
            return "", "UNION / set operations are not allowed."
        low = sql.lower().lstrip()
        try:
            if low.startswith("insert into"):
                return self._handle_insert(sql, env)
            rows = self.raw.execute(sql).fetchall()
            cols = [c[0] for c in self.raw.description]
            if not rows:
                return ("(0 rows) — tip: use ILIKE '%...%' for case-insensitive partial matches, "
                        "then try again.", None)
            return " | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows[:50]), None
        except Exception as ex:
            return "", f"{type(ex).__name__}: {ex}"

    def _handle_insert(self, sql: str, env) -> tuple[str, str | None]:
        m = re.match(r"insert\s+into\s+([a-zA-Z0-9_]+)", sql, re.I)
        sink = m.group(1) if m else None
        if sink in self.value_sinks:
            return self._insert_value_sink(sink, sql, env)
        if sink in self.url_sinks:
            return self._insert_url_sink(sink, sql, env)
        allowed = list(self.value_sinks) + list(self.url_sinks)
        return "", f"execute_intent only performs protected actions {allowed}. Use normal tools otherwise."

    def _insert_value_sink(self, sink, sql, env) -> tuple[str, str | None]:
        tool, _ = self.value_sinks[sink]
        self.raw.execute(f"DELETE FROM {sink}")
        try:  # Passant grounding, fail closed
            sql = self.conn.transform_query(sql)
        except Exception as ex:
            return "", f"blocked by the data-governance layer ({type(ex).__name__})"
        self.raw.execute(sql)
        staged = self.raw.execute(f"SELECT * FROM {sink}").fetchall()
        cols = [c[0] for c in self.raw.description]
        if not staged:
            return ("Blocked: the value is not grounded in your trusted data (0 rows). "
                    "The value must come from `trusted`.", None)
        return self._drain(env, tool, [dict(zip(cols, r)) for r in staged])

    def _insert_url_sink(self, sink, sql, env) -> tuple[str, str | None]:
        tool, _ = self.url_sinks[sink]
        self.raw.execute(f"DELETE FROM {sink}")
        self.raw.execute(sql)  # no Passant; we enforce the domain allowlist here
        staged = self.raw.execute(f"SELECT * FROM {sink}").fetchall()
        cols = [c[0] for c in self.raw.description]
        rows = [dict(zip(cols, r)) for r in staged]
        kept = [r for r in rows if _domain(str(r.get("url", ""))) in self.trusted_domains]
        if not kept:
            return (f"Blocked: '{rows[0].get('url') if rows else ''}' is not a trusted URL. Only "
                    "domains from your prompt, files, or contacts' emails may be visited.", None)
        return self._drain(env, tool, kept)

    def _drain(self, env, tool: str, rows: list[dict]) -> tuple[str, str | None]:
        outs = []
        if tool == "send_email":
            recipients = [r["recipient"] for r in rows if r.get("recipient")]
            first = rows[0]
            res, err = self.real_runtime.run_function(
                env, "send_email",
                {"recipients": recipients, "subject": first.get("subject") or "", "body": first.get("body") or ""})
            outs.append(err or str(res))
        elif tool == "input_to_webpage":
            for r in rows:
                args = {"url": r.get("url"), "input_ids": _to_list(r.get("input_ids")),
                        "input_values": _to_list(r.get("input_values"))}
                res, err = self.real_runtime.run_function(env, tool, args)
                outs.append(err or str(res))
        else:
            for r in rows:
                args = {k: v for k, v in r.items() if v is not None}
                res, err = self.real_runtime.run_function(env, tool, args)
                outs.append(err or str(res))
        return f"{tool} executed ({len(rows)} row(s)):\n" + "\n".join(o[:300] for o in outs[:5]), None
