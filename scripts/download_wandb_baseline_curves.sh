#!/usr/bin/env bash
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer

PYTHON=/home/vn-user0101/.conda/envs/htp/bin/python
AUDIT=paper_artifacts/wandb_baseline_curves/run_audit.json
DATA=paper_artifacts/wandb_baseline_curves/data
PLOTS=paper_artifacts/wandb_baseline_curves/plots
LOGS=paper_artifacts/wandb_baseline_curves/download_logs
mkdir -p "$LOGS"

# The frozen audit currently contains 177 selected runs: 176 finished and one
# explicitly skipped crashed Harmony run. Each process owns one cache file.
seq 0 176 | xargs -P 8 -I '{}' bash -c \
  '"$0" -m corewm_eval.wandb_baseline_curves download-one --audit "$1" --output-dir "$2" --index "$3" > "$4/$3.log" 2>&1' \
  "$PYTHON" "$AUDIT" "$DATA" '{}' "$LOGS"

# This pass is cache-only in practice and produces the consolidated raw tables.
"$PYTHON" -m corewm_eval.wandb_baseline_curves download \
  --audit "$AUDIT" --output-dir "$DATA" --workers 1
MPLCONFIGDIR=/tmp/corewm-matplotlib "$PYTHON" -m corewm_eval.wandb_baseline_curves plot \
  --data "$DATA/raw_episode_returns.parquet" --audit "$AUDIT" --output-dir "$PLOTS"
