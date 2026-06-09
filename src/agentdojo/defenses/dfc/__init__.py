"""DFC (Passant) defense — grounds protected sinks via model-authored SQL enforced by Passant.

Hybrid gating: only the guarded sinks (money/email/password/url) are routed through
execute_intent -> DFC -> tool; all other tools remain normal tool calls. See devnotes/INTEGRATION_NOTES.md."""

from agentdojo.defenses.dfc.bootstrap import DFCBootstrap
from agentdojo.defenses.dfc.executor import DFCToolsExecutor
from agentdojo.defenses.dfc.session import DFCSession

__all__ = ["DFCBootstrap", "DFCToolsExecutor", "DFCSession"]
