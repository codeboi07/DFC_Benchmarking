#!/bin/bash
# Smoke test Claude Sonnet 4.6 via Bedrock: baseline agent loop + DFC policy generation.
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="-s workspace --model CLAUDE_SONNET_4_6 --attack important_instructions -ut user_task_0 -it injection_task_0 -f"

echo "=== [1/2] BASELINE smoke (Bedrock agent loop) ==="
$PY -m agentdojo.scripts.benchmark $COMMON 2>&1 | tail -6

echo ""; echo "=== [2/2] DFC smoke (Bedrock agent + Bedrock policy generation) ==="
$PY -m agentdojo.scripts.benchmark $COMMON --defense dfc_agent_framework_integration --dfc-model claude-sonnet-4-6 2>&1 | tail -6
echo "=== SMOKE DONE ==="
