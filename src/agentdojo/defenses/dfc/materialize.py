"""DFC materializer (shopping) — builds the `trusted` relation and the guarded `*_input`
staging tables in DuckDB.

Hybrid design: only the GUARDED sinks are routed through SQL/DFC. The model reads ordinary
state via the normal suite tools, so we do NOT mirror the whole environment — we only need:
  - `trusted(value, owner, kind)` : USER-originated values a protected field may be grounded to
  - one `<tool>_input` staging table per guarded sink

`trusted` IS the security boundary. It is built ONLY from user-originated sources (the prompt,
the user's /user/info files, the user's own contacts) and deliberately EXCLUDES injection
channels (inbox email bodies, web/product content, downloaded files)."""

import re
from pathlib import Path

IBAN = re.compile(r"[A-Z]{2}[0-9]{6,}")
EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

# guarded staging table -> (real tool name, DuckDB column DDL). Columns mirror the tool's params;
# send_email is per-recipient (one row each), reassembled into a list on drain.
GUARDED_SINKS: dict[str, tuple[str, str]] = {
    "send_money_input": ("send_money", '("recipient" TEXT, "amount" DOUBLE, "subject" TEXT, "date" TEXT)'),
    "send_email_input": ("send_email", '("recipient" TEXT, "subject" TEXT, "body" TEXT)'),
    "update_shopping_account_password_input": ("update_shopping_account_password", '("password" TEXT)'),
    "download_file_through_url_input": ("download_file_through_url", '("url" TEXT, "save_dir" TEXT)'),
}
GUARDED_TOOLS = {tool for tool, _ in GUARDED_SINKS.values()}

# Prototype URL gating (defense="dfc_url"): also route the web tools through the gate and allow only
# URLs whose domain is in `trusted_urls`. ⚠️ KILLED EXPERIMENT — retained as a reproducible negative
# result; it does NOT affect the `dfc` defense (gate_urls defaults False). The URL allowlist is
# self-certifying: the injection lands in a contact's email, so the attacker URL leaks into
# `trusted_urls`. See devnotes/url_policy_prototype_spec.md §9 for the go/kill verdict.
URL_GUARDED_SINKS: dict[str, tuple[str, str]] = {
    "browse_webpage_input": ("browse_webpage", '("url" TEXT)'),
    "input_to_webpage_input": ("input_to_webpage", '("url" TEXT, "input_ids" TEXT, "input_values" TEXT)'),
}
URL_GUARDED_TOOLS = {tool for tool, _ in URL_GUARDED_SINKS.values()}

# Require an explicit www./http prefix so we extract URLs, not bare email domains.
URL = re.compile(r"(?:https?://|www\.)[\w\-.]+(?:/[\w\-./?=&%#]*)?")


def _domain(u: str) -> str:
    """Normalize a URL to a bare domain: strip scheme, leading www., and any path."""
    u = u.strip().lower().rstrip(".,)!\"'")
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.split("/")[0]


def build_trusted_urls(prompt: str, env) -> list[tuple[str]]:
    """trusted_urls(value) = domains the user can vouch for: URLs in the prompt, the user's
    /user/info files, and emails sent by ADDRESS-BOOK CONTACTS. This deliberately relaxes the
    'exclude emails' boundary to a sender allowlist (a heuristic, not provenance — see spec)."""
    domains: set[str] = set()
    for u in URL.findall(prompt):
        domains.add(_domain(u))
    for _path, content in _userinfo_files(env):
        for u in URL.findall(content):
            domains.add(_domain(u))
    # URLs from emails whose sender is a known contact
    try:
        edb = env.email_database.model_dump()
        cur = edb.get("current_account")
        contacts = {c.get("email") for _ib in edb.get("inbox_list", [])
                    if _ib.get("account_email") == cur for c in (_ib.get("contact_list") or [])}
        for ib in edb.get("inbox_list", []):
            if ib.get("account_email") != cur:
                continue
            emails = []
            for key in ("received", "emails"):
                v = ib.get(key)
                emails += list(v.values()) if isinstance(v, dict) else (v or [])
            for e in emails:
                if e.get("sender") in contacts:
                    for u in URL.findall(str(e.get("body", ""))):
                        domains.add(_domain(u))
    except Exception:
        pass
    return [(d,) for d in sorted(domains) if d]


def _kind(v: str) -> str:
    if IBAN.fullmatch(v):
        return "bank account"
    if EMAIL.fullmatch(v):
        return "email address"
    return "value"


def _userinfo_files(env) -> list[tuple[str, str]]:
    """Read /user/info/* straight from the filesystem tree (no tool side effects)."""
    try:
        node = env.filesystem.model_dump()["root"]
        for part in ("user", "info"):
            node = (node.get("children") or {}).get(part)
            if node is None:
                return []
        return [
            (f"/user/info/{name}", str(child.get("content", "")))
            for name, child in (node.get("children") or {}).items()
            if child.get("type") == "file"
        ]
    except Exception:
        return []


def _contacts(env) -> list[tuple[str, str]]:
    """(email, name) from the current account's address book — legitimate email recipients."""
    out = []
    try:
        edb = env.email_database.model_dump()
        cur = edb.get("current_account")
        for ib in edb.get("inbox_list", []):
            if ib.get("account_email") == cur:
                for c in ib.get("contact_list", []) or []:
                    if c.get("email"):
                        out.append((c["email"], c.get("name", "contact")))
    except Exception:
        pass
    return out


def build_trusted(prompt: str, env) -> list[tuple[str, str, str]]:
    """Trusted = (value, owner, kind), USER-originated only. See module docstring."""
    out: list[tuple[str, str, str]] = []
    for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt):  # quoted prompt tokens
        tok = a or b
        if tok and "'" not in tok and '"' not in tok:
            out.append((tok, "you (the user)", _kind(tok)))
    for path, content in _userinfo_files(env):
        m = re.search(r"about ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)", content)
        owner = m.group(1) if m else Path(path).stem
        for v in set(IBAN.findall(content)) | set(EMAIL.findall(content)):
            out.append((v, owner, _kind(v)))
    for email, name in _contacts(env):
        out.append((email, name, "email address"))
    seen, uniq = set(), []
    for row in out:
        if row not in seen:
            seen.add(row); uniq.append(row)
    return uniq


def create_relations(raw, prompt: str, env, gate_urls: bool = False) -> None:
    """(Re)build trusted + guarded staging tables on the raw DuckDB connection.
    If gate_urls, also build trusted_urls + the web-tool staging tables."""
    trusted = build_trusted(prompt, env)
    raw.execute("CREATE OR REPLACE TABLE trusted(value TEXT, owner TEXT, kind TEXT)")
    if trusted:
        raw.executemany("INSERT INTO trusted VALUES (?,?,?)", trusted)
    for sink, (_tool, ddl) in GUARDED_SINKS.items():
        raw.execute(f"CREATE OR REPLACE TABLE {sink} {ddl}")
    if gate_urls:
        urls = build_trusted_urls(prompt, env)
        raw.execute("CREATE OR REPLACE TABLE trusted_urls(value TEXT)")
        if urls:
            raw.executemany("INSERT INTO trusted_urls VALUES (?)", urls)
        for sink, (_tool, ddl) in URL_GUARDED_SINKS.items():
            raw.execute(f"CREATE OR REPLACE TABLE {sink} {ddl}")
