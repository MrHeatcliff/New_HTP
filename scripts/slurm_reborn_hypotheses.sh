#!/usr/bin/env bash
#SBATCH --job-name=reborn-hypotheses
#SBATCH --partition=gpu_general
#SBATCH --qos=gpu_general_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-hypotheses-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-hypotheses-%j.err
set -euo pipefail
readonly RUNROOT="${1:?Frozen run root}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$RUNROOT/source"
exec /home/vn-user0101/Dat/HTS-Dreamer/.venv/bin/python -u -m corewm_eval.reborn_hypothesis_suite "$RUNROOT"
