#!/bin/bash
# Full workspace run on Claude Sonnet 4.6 via Bedrock. Usage: run_workspace_bedrock.sh {baseline|dfc}
# Attempt 1 uses -f (fresh). On crash/throttle, retries WITHOUT -f (resume, skips completed pairs).
cd "$(dirname "$0")"
PY=.venv/bin/python
ARM="$1"
BASE="-s workspace --model CLAUDE_SONNET_4_6 --attack important_instructions"

if [ "$ARM" = "dfc" ]; then
    EXTRA="--defense dfc_agent_framework_integration --dfc-model claude-sonnet-4-6"
    LOG="full_bedrock_dfc.log"
elif [ "$ARM" = "baseline" ]; then
    EXTRA=""
    LOG="full_bedrock_baseline.log"
else
    echo "usage: $0 {baseline|dfc}"; exit 2
fi
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MAX=10
for attempt in $(seq 1 $MAX); do
    if [ "$attempt" -eq 1 ]; then FLAG="-f"; else FLAG=""; fi
    log "=== bedrock $ARM attempt $attempt/$MAX (flag='$FLAG') ==="
    $PY -m agentdojo.scripts.benchmark $BASE $EXTRA $FLAG >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "=== bedrock $ARM COMPLETE (exit 0) ==="
        exit 0
    fi
    log "=== bedrock $ARM attempt $attempt failed (exit $rc); waiting 60s then resuming ==="
    sleep 60
done
log "=== bedrock $ARM GAVE UP after $MAX attempts ==="
exit 1
