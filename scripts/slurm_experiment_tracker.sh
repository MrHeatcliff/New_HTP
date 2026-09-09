#!/usr/bin/env bash
# Run one row from the experiment workbook on an isolated H200 allocation.
# The launcher derives the workbook Run ID and owns resume semantics.
#
# Usage:
#   sbatch scripts/slurm_experiment_tracker.sh htp_full 0 atari100k_breakout size25m

#SBATCH --job-name=htp-tracker
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/htp-tracker-%j.out
#SBATCH --error=slurm_logs/htp-tracker-%j.err

set -euo pipefail

readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
cd "$REPO"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
# Compute nodes on this cluster do not currently resolve api.wandb.ai. Keep
# training independent of that network path; `wandb sync` can upload later.
export WANDB_MODE="${WANDB_MODE:-offline}"
export HTP_GPU_COUNT="${HTP_GPU_COUNT:-2}"

exec ./run_arm.sh "$@"
