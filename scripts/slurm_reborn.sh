#!/usr/bin/env bash
#SBATCH --job-name=reborn-four-game
#SBATCH --partition=gpu_general
#SBATCH --qos=gpu_general_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/reborn-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
readonly RUNROOT="${1:?Frozen run root}"
readonly PYTHON=/home/vn-user0101/Dat/HTS-Dreamer/.venv/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$RUNROOT/source"
# All JAX tests and training happen inside this allocation.
"$PYTHON" -m pytest tests/test_reborn.py tests/test_h200_suite.py -q > "$RUNROOT/tests.log" 2>&1
exec "$PYTHON" -u -m corewm_eval.reborn_experiment "$RUNROOT"
