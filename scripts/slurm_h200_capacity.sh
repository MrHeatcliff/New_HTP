#!/usr/bin/env bash
#SBATCH --job-name=h200-capacity
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/h200-capacity-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/h200-capacity-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
cd /home/vn-user0101/Dat/HTS-Dreamer
exec .venv/bin/python -u -m corewm_eval.h200_capacity "paper_artifacts/persistence_research/h200_capacity_${SLURM_JOB_ID}"
