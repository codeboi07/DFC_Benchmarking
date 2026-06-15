#!/bin/bash
# Vulnerability sweep: which attacks does Claude Sonnet 4.6 actually fall for?
# Baseline only (no DFC), 3 user tasks x 3 injection tasks each. -f for fresh.
cd "$(dirname "$0")"
PY=.venv/bin/python
ATTACKS="direct ignore_previous system_message injecagent tool_knowledge dos"
UT="-ut user_task_0 -ut user_task_1 -ut user_task_2"
IT="-it injection_task_0 -it injection_task_1 -it injection_task_2"
for attack in $ATTACKS; do
  echo "=== Probing attack: $attack ==="
  $PY -m agentdojo.scripts.benchmark -s workspace --model BEDROCK_CLAUDE_SONNET_4_6 \
      --attack "$attack" $UT $IT -f >> probe_attacks.log 2>&1
  echo "  done: $attack (exit $?)"
done
echo "=== PROBE DONE ==="
