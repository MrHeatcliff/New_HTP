#!/usr/bin/env bash
#SBATCH --job-name=harmony-five-seed-report
#SBATCH --partition=cpu_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/harmony-five-seed-report-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/harmony-five-seed-report-%j.err
set -euo pipefail
cd /home/vn-user0101/Dat/HTS-Dreamer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/vn-user0101/.conda/envs/htp/bin/python -m corewm_eval.harmony_five_seed_report
