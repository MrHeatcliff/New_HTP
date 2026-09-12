#!/usr/bin/env bash
#SBATCH --job-name=reborn-checkpoint-eval
#SBATCH --partition=gpu_general
#SBATCH --qos=gpu_general_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-checkpoint-eval-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-checkpoint-eval-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u -m corewm_eval.reborn_checkpoint_eval \
  production_runs/reborn_four_game_fx3t85mi \
  production_runs/reborn_checkpoint_evaluation_v1
