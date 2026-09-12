#!/usr/bin/env bash
#SBATCH --job-name=reborn-eval-figures
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-eval-figures-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-eval-figures-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
exec /home/vn-user0101/.conda/envs/htp/bin/python -m corewm_eval.reborn_checkpoint_report \
  production_runs/reborn_checkpoint_evaluation_v1 \
  paper_artifacts/reborn_checkpoint_evaluation
