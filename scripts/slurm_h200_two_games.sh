#!/usr/bin/env bash
# Usage: sbatch scripts/slurm_h200_two_games.sh [actions-per-game]
# Same tested size25m configuration, Alien + Breakout, fresh seed 0 runs.
# Default is a short run; explicitly pass 110000 for long training.
#SBATCH --job-name=h200-two-games
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/h200-two-games-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/h200-two-games-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
cd /home/vn-user0101/Dat/HTS-Dreamer
export HTP_CAPACITY_PAIRED_ONLY=1
export HTP_CAPACITY_ACTIONS="${1:-5000}"
export HTP_CAPACITY_GATE=paper_artifacts/persistence_research/h200_capacity_3334/results.json
exec .venv/bin/python -u -m corewm_eval.h200_capacity "production_runs/h200_shared_${SLURM_JOB_ID}"
