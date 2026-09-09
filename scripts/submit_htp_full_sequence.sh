#!/usr/bin/env bash
# Submit the remaining no-warmup htp_full rows from the experiment workbook.
# Slurm itself acts as the durable supervisor through afterok dependencies:
#   run_007 (seed 0) -> run_008 (seed 1) -> run_009 (seed 2)
# A successor starts only when its predecessor exits successfully.

set -euo pipefail

readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly RUNNER="$REPO/scripts/slurm_experiment_tracker.sh"
readonly CONTROL="$REPO/production_runs/experiment_tracker/full_sequence"
readonly LEDGER="$CONTROL/submission_ledger.tsv"
readonly LOCK="$CONTROL/submission.lock"

CURRENT_JOB_ID=${1:?usage: $0 CURRENT_RUN_007_JOB_ID}

mkdir -p "$CONTROL"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another full-sequence submitter is active" >&2
  exit 2
fi
if [[ -s "$LEDGER" ]]; then
  echo "ERROR: full-sequence ledger already exists: $LEDGER" >&2
  echo "Refusing to submit duplicate jobs." >&2
  exit 3
fi

CURRENT_ROW=$(squeue -h -j "$CURRENT_JOB_ID" -o '%A|%j|%T')
if [[ "$CURRENT_ROW" != "$CURRENT_JOB_ID|run007-htp-full-2gpu|RUNNING" ]]; then
  echo "ERROR: job $CURRENT_JOB_ID is not the active run_007 2-GPU job" >&2
  echo "Observed: ${CURRENT_ROW:-<not in queue>}" >&2
  exit 4
fi

printf 'run_id\tseed\tslurm_job_id\tdepends_on\n' > "$LEDGER"
printf 'run_007\t0\t%s\t\n' "$CURRENT_JOB_ID" >> "$LEDGER"

PREVIOUS_JOB_ID=$CURRENT_JOB_ID
for SEED in 1 2; do
  printf -v RUN_NUMBER '%03d' "$((7 + SEED))"
  RUN_ID="run_${RUN_NUMBER}"
  JOB_ID=$(sbatch --parsable \
    --job-name="${RUN_ID}-htp-full-2gpu" \
    --dependency="afterok:${PREVIOUS_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    "$RUNNER" htp_full "$SEED" atari100k_breakout size25m)
  JOB_ID=${JOB_ID%%;*}
  printf '%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$SEED" "$JOB_ID" "$PREVIOUS_JOB_ID" >> "$LEDGER"
  echo "Submitted $RUN_ID as job $JOB_ID afterok:$PREVIOUS_JOB_ID"
  PREVIOUS_JOB_ID=$JOB_ID
done

echo "Full HTP sequence is supervised by Slurm dependencies."
echo "Ledger: $LEDGER"
