#!/usr/bin/env bash
#SBATCH --partition=gpu_trial
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=production_runs/corewm_atari100k_v1/slurm_logs/%x-%j.out
#SBATCH --error=production_runs/corewm_atari100k_v1/slurm_logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch $0 METHOD OUTPUT_SLUG" >&2
  exit 2
fi

METHOD="$1"
OUTPUT_SLUG="$2"
REPO=/home/vn-user0101/Dat/HTS-Dreamer
DATASET="$REPO/production_runs/corewm_atari100k_v1/shared_diagnostic_trajectories/alien/seed_0"
CACHE="$REPO/production_runs/corewm_atari100k_v1/offline_cache/alien/seed_0/$METHOD"
OUTPUT="$REPO/production_runs/corewm_atari100k_v1/offline_evaluation/alien/seed_0/posthoc_reconstruction/$OUTPUT_SLUG"

cd "$REPO"
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run --no-capture-output -n htp python -m corewm_eval.posthoc_train \
  --method "$METHOD" \
  --cache "$CACHE" \
  --dataset "$DATASET" \
  --output "$OUTPUT"
