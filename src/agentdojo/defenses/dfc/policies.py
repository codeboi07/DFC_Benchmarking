"""Loads DFC policies for a suite. Policy artifacts are authored by the defender and live under
the repo-root `policies/` directory (e.g. `policies/shopping_policies.py`), kept separate from
the package so they are clearly the human-authored security spec."""

import importlib.util
from pathlib import Path

from data_flow_control import Policy

# repo root = .../<root>/src/agentdojo/defenses/dfc/policies.py -> parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]

_POLICY_FILES = {
    "shopping": ("shopping_policies", "shopping_policies"),  # (module file stem, factory fn)
}


def get_policies(suite_name: str | None) -> list[Policy]:
    """Return the registered DFC policies for `suite_name` ([] if none authored)."""
    if suite_name not in _POLICY_FILES:
        return []
    stem, factory = _POLICY_FILES[suite_name]
    path = _REPO_ROOT / "policies" / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"DFC policy file not found for suite '{suite_name}': {path}")
    spec = importlib.util.spec_from_file_location(f"dfc_policies_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return list(getattr(module, factory)())
