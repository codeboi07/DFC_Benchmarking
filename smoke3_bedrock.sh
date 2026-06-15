#!/bin/bash
# 3-task smoke after date + policy-gen fixes. Claude Sonnet 4.6 via Bedrock.
cd "$(dirname "$0")"
PY=.venv/bin/python
UT="-ut user_task_0 -ut user_task_1 -ut user_task_2"
COMMON="-s workspace --model CLAUDE_SONNET_4_6 --attack important_instructions $UT -it injection_task_0 -f"

echo "=== BASELINE (3 tasks, Bedrock) ==="
$PY -m agentdojo.scripts.benchmark $COMMON 2>&1 | tail -5

echo ""; echo "=== DFC (3 tasks, Bedrock agent + Bedrock policy gen) ==="
$PY -m agentdojo.scripts.benchmark $COMMON --defense dfc_agent_framework_integration --dfc-model claude-sonnet-4-6 2>&1 | tail -5
echo "=== SMOKE3 DONE ==="
