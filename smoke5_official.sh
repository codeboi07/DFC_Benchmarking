#!/bin/bash
# 5-task smoke on the OFFICIAL admission-control build. Claude Sonnet 4.6 via Bedrock ARN.
# Key test: user_task_9 / user_task_12 (write tasks) false positives should disappear.
cd "$(dirname "$0")"
PY=.venv/bin/python
UT="-ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_9 -ut user_task_12"
COMMON="-s workspace --model BEDROCK_CLAUDE_SONNET_4_6 --attack important_instructions $UT -it injection_task_0 -f"

echo "=== BASELINE (5 tasks, official build) ==="
$PY -m agentdojo.scripts.benchmark $COMMON 2>&1 | tail -5

echo ""; echo "=== DFC (5 tasks, official admission control) ==="
$PY -m agentdojo.scripts.benchmark $COMMON --defense dfc_agent_framework_integration 2>&1 | tail -5
echo "=== SMOKE5 DONE ==="
