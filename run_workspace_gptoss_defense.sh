#!/bin/bash
# Full workspace run for GPT-OSS 120B under a built-in defense. Usage: run_workspace_gptoss_defense.sh <defense>
# No -f (no prior results). On crash/throttle, retries (already-completed pairs are skipped on rerun).
cd "$(dirname "$0")"
PY=.venv/bin/python
DEF="$1"
[ -z "$DEF" ] && { echo "usage: $0 <defense>"; exit 2; }
LOG="full_gptoss_${DEF}.log"
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MAX=12
for attempt in $(seq 1 $MAX); do
    log "=== gptoss $DEF attempt $attempt/$MAX (no -f) ==="
    $PY -m agentdojo.scripts.benchmark -s workspace --model BEDROCK_GPT_OSS_120B \
        --attack important_instructions --defense "$DEF" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "=== gptoss $DEF COMPLETE (exit 0) ==="
        exit 0
    fi
    log "=== gptoss $DEF attempt $attempt failed (exit $rc); waiting 60s then resuming ==="
    sleep 60
done
log "=== gptoss $DEF GAVE UP after $MAX attempts ==="
exit 1
