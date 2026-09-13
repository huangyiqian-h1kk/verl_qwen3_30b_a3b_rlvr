#!/usr/bin/env bash
# Start from the repository: bash monitor_rg_16k.sh
# Take over the recorded job for another 24 hours (no initial qsub):
#   RG_MAX_RUNTIME_SECONDS=86400 bash monitor_rg_16k.sh resume
# Stop this monitor (leave the PBS job alone): bash monitor_rg_16k.sh stop
# Defaults: check every 20 minutes, monitor for 12 hours, retry only exit 271.
set -euo pipefail
export LC_ALL=C
umask 077

REPO_ROOT="${RG_REPO_ROOT:-/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr}"
STATE_DIR="${RG_MONITOR_DIR:-$REPO_ROOT/logs/rg_16k_monitor}"
JOB_SCRIPT="jobs/0390_q3_rg_sixcat_r08_16k_from25_full500_k8.pbs"
CHECK_INTERVAL="${RG_CHECK_INTERVAL_SECONDS:-1200}"
MAX_RUNTIME="${RG_MAX_RUNTIME_SECONDS:-43200}"
STOP_POLL="${RG_STOP_POLL_SECONDS:-5}"
STOP_FILE="$STATE_DIR/stop.request"
JOB_ID_FILE="$STATE_DIR/current_job.id"
SUBMIT_ARGS=(-v RTYPE=rt_HF,ACTOR_LR=1e-6,RUN_VERSION=v1_len16k)
ACTION="${1:-start}"

log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }

if (( $# > 1 )); then
    printf 'Usage: bash %s [start|resume|stop]\n' "$0" >&2
    exit 2
fi
case "$ACTION" in
    stop)
        mkdir -p "$STATE_DIR"
        : > "$STOP_FILE"
        log "Stop requested. The monitor will exit; the PBS job will NOT be cancelled."
        exit 0
        ;;
    start|resume) ;;
    *) printf 'Usage: bash %s [start|resume|stop]\n' "$0" >&2; exit 2 ;;
esac

for value in "$CHECK_INTERVAL" "$MAX_RUNTIME" "$STOP_POLL"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR: interval, runtime and stop-poll settings must be positive integer seconds."
        exit 2
    fi
done
for command_name in qsub qstat flock; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        log "ERROR: command not found: $command_name"
        exit 1
    fi
done
cd "$REPO_ROOT"
if [[ ! -f "$JOB_SCRIPT" ]]; then
    log "ERROR: PBS script not found: $REPO_ROOT/$JOB_SCRIPT"
    exit 1
fi
mkdir -p "$STATE_DIR"

# A kernel lock prevents two monitors using this directory from submitting jobs.
# Do not delete the lock file: its existence does not mean the lock is held.
exec 9>"$STATE_DIR/monitor.lock"
if [[ "$ACTION" == "resume" ]]; then
    if [[ ! -s "$JOB_ID_FILE" ]]; then
        log "ERROR: no recorded job ID in $JOB_ID_FILE. Nothing submitted."
        exit 1
    fi
    # Works with the original monitor too: ask it to stop and wait for its lock.
    # Read the final job ID only AFTER it exits, in case it was submitting a retry.
    : > "$STOP_FILE"
    log "Requesting monitor handover; waiting up to 60 seconds for its lock."
    if ! flock -w 60 9; then
        log "ERROR: handover timed out; nothing submitted. The stop request remains pending."
        exit 1
    fi
else
    if ! flock -n 9; then
        log "ERROR: another monitor holds $STATE_DIR/monitor.lock; nothing submitted."
        exit 1
    fi
fi
rm -f "$STOP_FILE"

job_id=""
submission_count=0
deadline=$((SECONDS + MAX_RUNTIME))
cleanup() {
    log "Monitor stopped. Last tracked job: ${job_id:-none}. No qdel was issued."
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

should_stop() {
    if [[ -e "$STOP_FILE" ]]; then
        log "Manual stop requested."
        return 0
    fi
    if (( SECONDS >= deadline )); then
        log "Monitoring time limit reached ($MAX_RUNTIME seconds)."
        return 0
    fi
    return 1
}

submit_job() {
    local submission
    if should_stop; then exit 0; fi
    # Keep stderr visible. Capture the job ID directly from qsub stdout.
    # Close the lock descriptor in child commands to avoid inherited locks.
    if ! submission=$(qsub "${SUBMIT_ARGS[@]}" "$JOB_SCRIPT" 9>&-); then
        log "ERROR: qsub failed. Stopping; check qstat before submitting manually."
        exit 1
    fi
    if [[ "$submission" =~ ^[[:space:]]*([0-9]+(\.[A-Za-z0-9._-]+)?)[[:space:]]*$ ]]; then
        job_id="${BASH_REMATCH[1]}"
    else
        log "ERROR: cannot identify qsub job ID from: $submission"
        log "A job may have been accepted. Stopping without another submission; check qstat."
        exit 1
    fi
    submission_count=$((submission_count + 1))
    printf '%s\n' "$job_id" > "$JOB_ID_FILE"
    log "Submitted $job_id (submission $submission_count)."
}

wait_for_check() {
    local next_check delay
    next_check=$((SECONDS + CHECK_INTERVAL))
    if (( next_check > deadline )); then next_check=$deadline; fi
    while (( SECONDS < next_check )); do
        if should_stop; then return 1; fi
        delay=$((next_check - SECONDS))
        if (( delay <= 0 )); then break; fi
        if (( delay > STOP_POLL )); then delay=$STOP_POLL; fi
        sleep "$delay" 9>&-
    done
    if should_stop; then return 1; fi
    return 0
}

log "Starting: interval=${CHECK_INTERVAL}s; total runtime=${MAX_RUNTIME}s."
log "Stop command: bash monitor_rg_16k.sh stop"
if [[ "$ACTION" == "resume" ]]; then
    recorded_id=$(< "$JOB_ID_FILE")
    if [[ "$recorded_id" =~ ^[[:space:]]*([0-9]+(\.[A-Za-z0-9._-]+)?)[[:space:]]*$ ]]; then
        job_id="${BASH_REMATCH[1]}"
        log "Resumed monitoring existing job $job_id; no initial qsub. Timer restarted."
    else
        log "ERROR: invalid recorded job ID in $JOB_ID_FILE; nothing submitted."
        exit 1
    fi
else
    submit_job
fi

while wait_for_check; do
    # -f gives full attributes; -x includes finished jobs retained in PBS history.
    # A query failure / missing history is not proof that the job exited 271.
    if ! details=$(qstat -fx "$job_id" 9>&- 2>&1); then
        log "WARNING: qstat failed for $job_id; no resubmission. ${details//$'\n'/ }"
        continue
    fi
    job_state=$(awk '$1 == "job_state" && $2 == "=" {print $3; exit}' <<< "$details")
    exit_status=$(awk '$1 == "Exit_status" && $2 == "=" {print $3; exit}' <<< "$details")
    log "Checked $job_id: job_state=${job_state:-unknown}, Exit_status=${exit_status:-not available}."
    if should_stop; then break; fi

    case "$job_state" in
        F|C)
            if [[ "$exit_status" == "271" ]]; then
                log "Confirmed finished with Exit_status=271; resubmitting."
                submit_job
            elif [[ "$exit_status" =~ ^-?[0-9]+$ ]]; then
                log "Finished with Exit_status=$exit_status. No retry; monitoring complete."
                break
            else
                log "WARNING: finished but Exit_status is missing; wait for the next check."
            fi
            ;;
        Q|R|H|S|E|W|T|B|M|U)
            # Queued, held, suspended and exiting jobs must not be duplicated.
            ;;
        *) log "WARNING: unrecognised/missing job state; no resubmission." ;;
    esac
done
