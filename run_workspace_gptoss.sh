#!/bin/bash
# Full workspace run: GPT-OSS 120B agent (Bedrock Converse). Usage: run_workspace_gptoss.sh {baseline|dfc}
# DFC arm uses Opus 4.1 for policy generation. Attempt 1 uses -f (fresh); on crash/throttle it retries
# WITHOUT -f (resume — skips completed pairs).
cd "$(dirname "$0")"
PY=.venv/bin/python
ARM="$1"
OPUS='arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/1qmjorj3itx8'
BASE="-s workspace --model BEDROCK_GPT_OSS_120B --attack important_instructions"

if [ "$ARM" = "dfc" ]; then
    EXTRA="--defense dfc_agent_framework_integration --dfc-model $OPUS"
    LOG="full_gptoss_dfc.log"
elif [ "$ARM" = "baseline" ]; then
    EXTRA=""
    LOG="full_gptoss_baseline.log"
else
    echo "usage: $0 {baseline|dfc}"; exit 2
fi
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

MAX=12
# Resume mode (no -f): skips already-completed pairs. Used so we can relaunch after a code fix
# without redoing finished work. The original run already created fresh results this session.
for attempt in $(seq 1 $MAX); do
    FLAG=""
    log "=== gptoss $ARM attempt $attempt/$MAX (resume, flag='$FLAG') ==="
    $PY -m agentdojo.scripts.benchmark $BASE $EXTRA $FLAG >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "=== gptoss $ARM COMPLETE (exit 0) ==="
        exit 0
    fi
    log "=== gptoss $ARM attempt $attempt failed (exit $rc); waiting 60s then resuming ==="
    sleep 60
done
log "=== gptoss $ARM GAVE UP after $MAX attempts ==="
exit 1
