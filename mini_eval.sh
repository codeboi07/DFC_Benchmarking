#!/bin/bash
# Mini characterization: 5 read-only workspace tasks x injection_task_0, DFC vs baseline.
cd "$(dirname "$0")"
PY=.venv/bin/python
UT="-ut user_task_0 -ut user_task_1 -ut user_task_2 -ut user_task_3 -ut user_task_4"
COMMON="-s workspace --model GPT_4O_2024_05_13 --attack important_instructions $UT -it injection_task_0 -f"

echo "=== BASELINE (no defense) ==="
$PY -m agentdojo.scripts.benchmark $COMMON 2>&1 | tail -8

echo ""; echo "=== DFC (dfc_model=gpt-4o-mini) ==="
$PY -m agentdojo.scripts.benchmark $COMMON --dfc-model gpt-4o-mini-2024-07-18 --defense dfc_agent_framework_integration 2>&1 | tail -8
echo "=== MINI EVAL DONE ==="
