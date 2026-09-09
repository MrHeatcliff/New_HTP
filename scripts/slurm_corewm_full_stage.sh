#!/usr/bin/env bash
# Full-only Stage 1 runner. Training itself executes from the frozen clean
# worktree supplied by the method-stage feeder.
#SBATCH --job-name=corewm-v1-full
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/corewm-v1-full-%j.out
#SBATCH --error=slurm_logs/corewm-v1-full-%j.err

set -euo pipefail

readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly PYTHON=/home/vn-user0101/.conda/envs/htp/bin/python
: "${COREWM_STAGE_ORDINAL:?COREWM_STAGE_ORDINAL must be set}"
: "${COREWM_STAGE_MANIFEST:?COREWM_STAGE_MANIFEST must be set}"
: "${COREWM_CLEAN_WORKTREE:?COREWM_CLEAN_WORKTREE must be set}"

cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" -m corewm_eval.method_stage run-job \
  --manifest "$COREWM_STAGE_MANIFEST" \
  --ordinal "$COREWM_STAGE_ORDINAL" \
  --worktree "$COREWM_CLEAN_WORKTREE"
