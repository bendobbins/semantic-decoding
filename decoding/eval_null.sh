#!/bin/bash
# Rerun evaluate_predictions.py over every S1 run and both test conditions with
# a configurable null count.
#
# The default --null 10 in evaluate_predictions.py estimates the z-score
# denominator from 10 samples, giving it ~24% relative error (1/sqrt(2(n-1))).
# That is large enough to dominate the run-to-run z-score differences. Raising
# the count shrinks it: ~14% at 25, ~10% at 50, ~7% at 100.
#
# Cost is linear in the null count -- generate_null() runs a full beam search
# per null sequence -- so time the sweep before committing to a large value.
#
# Usage:
#   ./eval_null.sh [NULL] [PARALLEL]
#     NULL      null sequences per job   (default 50)
#     PARALLEL  concurrent jobs          (default 1)
#
# NOTE: this overwrites <repo>/scores/<subject>/perceived_speech/<task>.npz.
# Move or copy that directory first if the existing scores still matter.

set -u

NULL=${1:-50}
PAR=${2:-1}
export NULL

cd "$(dirname "$0")" || exit 1
LOGDIR="../scores_eval_logs"
mkdir -p "$LOGDIR"
export LOGDIR

RUNS=${RUNS:-"S1 S1_small S1_noise_s0.25 S1_noise_s0.5 S1_noise_s0.7"}

# Entries are "task" or "task:reference". A reference is needed when the task
# name is not a key in data_test/eval_segments.json: the _noise100_s0.5 variant
# is the same stimulus with noise added to the *neural* data, so it scores
# against the wheretheressmoke transcript and its [10, 592] eval segment.
# Without it, evaluate_predictions.py raises KeyError at eval_segments[reference]
# -- and only after generate_null() has already burned a full beam search.
TASKS=${TASKS:-"wheretheressmoke wheretheressmoke_noise100_s0.5:wheretheressmoke"}

run_one() {
    run=$1
    spec=$2
    task=${spec%%:*}
    ref=${spec#*:}
    log="$LOGDIR/${run}_${task}.log"
    if [ "$ref" != "$spec" ]; then
        set -- --references "$ref"
    else
        set --
    fi
    start=$(date +%s)
    if python3 evaluate_predictions.py \
            --subject "$run" --experiment perceived_speech \
            --task "$task" --null "$NULL" "$@" >"$log" 2>&1; then
        echo "[done ] $run $task  ($(( $(date +%s) - start ))s)"
    else
        echo "[FAIL ] $run $task  -> $log"
    fi
}
export -f run_one

echo "null=$NULL  parallel=$PAR  jobs=$(( $(echo $RUNS | wc -w) * $(echo $TASKS | wc -w) ))"
started=$(date +%s)

for task in $TASKS; do
    for run in $RUNS; do
        printf '%s\0%s\0' "$run" "$task"
    done
done | xargs -0 -P "$PAR" -n 2 bash -c 'run_one "$@"' _

echo "all jobs finished in $(( ($(date +%s) - started) / 60 )) min"
