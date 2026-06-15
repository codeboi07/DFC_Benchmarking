#!/bin/bash
# Build an isolated venv for the DFC_Benchmarking fork.
# Order: install the local DFC integration package first (pulls data-flow-control),
# then the agentdojo fork editable (its unpinned dfc dep is then already satisfied).
set -e
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3

echo "[setup] creating .venv"
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel >/dev/null

echo "[setup] installing dfc_agent_framework_integration (+ data-flow-control)"
.venv/bin/pip install -e ./dfc_agent_framework_integration

echo "[setup] installing agentdojo fork (editable)"
.venv/bin/pip install -e .

echo "[setup] verifying imports"
.venv/bin/python -c "
import agentdojo, dfc_agent_framework_integration, data_flow_control
print('agentdojo from :', agentdojo.__file__)
print('dfc integration:', dfc_agent_framework_integration.__file__)
print('data_flow_control:', getattr(data_flow_control,'__version__','?'))
"
echo "[setup] DONE"
