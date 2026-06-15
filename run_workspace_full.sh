#!/bin/bash
# Resumable full workspace run. Usage: run_workspace_full.sh {baseline|dfc}
# First attempt uses -f (fresh). On non-zero exit (rate limit / crash), retries WITHOUT -f
# so already-completed (user_task x injection_task) pairs are skipped (resume from checkpoint).
cd "$(dirname "$0")"
PY=.venv/bin/python
ARM="$1"
BASE="-s workspace --model GPT_4O_2024_05_13 --attack important_instructions"

if [ "$ARM" = "dfc" ]; then
    EXTRA="--defense dfc_agent_framework_integration --dfc-model gpt-4o-2024-08-06"
    LOG="full_dfc.log"
elif [ "$ARM" = "baseline" ]; then
    EXTRA=""
    LOG="full_baseline.log"
else
    echo "usage: $0 {baseline|dfc}"; exit 2
fi
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MAX=8
# No -f: resume mode. Already-completed (user_task x injection) pairs are skipped, so this
# safely continues a partially-finished run (e.g. after an OpenAI quota top-up).
for attempt in $(seq 1 $MAX); do
    FLAG=""
    log "=== $ARM attempt $attempt/$MAX (resume, flag='$FLAG') ==="
    $PY -m agentdojo.scripts.benchmark $BASE $EXTRA $FLAG >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "=== $ARM COMPLETE (exit 0) ==="
        exit 0
    fi
    log "=== $ARM attempt $attempt failed (exit $rc); waiting 60s then resuming ==="
    sleep 60
done
log "=== $ARM GAVE UP after $MAX attempts ==="
exit 1
