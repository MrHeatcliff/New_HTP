#!/usr/bin/env bash
# Submit as: sbatch --array=0-35%2 scripts/slurm_corewm_wave1.sh
# Array task IDs are immutable job_index values in the frozen 680-run manifest.
#SBATCH --job-name=corewm-v1-w1
#SBATCH --partition=gpu_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/corewm-v1-w1-%A_%a.out
#SBATCH --error=slurm_logs/corewm-v1-w1-%A_%a.err

set -euo pipefail

readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly PYTHON=/home/vn-user0101/.conda/envs/htp/bin/python
readonly PROTOCOL="$REPO/production_runs/corewm_atari100k_v1/protocol.json"

cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DIR="$REPO/production_runs/corewm_atari100k_v1/wandb"
mkdir -p "$WANDB_DIR"

"$PYTHON" -m corewm_eval.production run-job \
  --protocol "$PROTOCOL" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --attempt 1
