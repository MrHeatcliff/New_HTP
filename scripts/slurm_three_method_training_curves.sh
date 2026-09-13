#!/usr/bin/env bash
#SBATCH --job-name=three-training-curves
#SBATCH --partition=cpu_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/three-training-curves-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/three-training-curves-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/vn-user0101/.conda/envs/htp/bin/python -m corewm_eval.three_method_training_curves
