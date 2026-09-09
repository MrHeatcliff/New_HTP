#!/usr/bin/env bash
#SBATCH --job-name=constraint-full26
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/full26-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/full26-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
readonly ROOT="${1:?Supply frozen submission directory}"
readonly PYTHON=/home/vn-user0101/Dat/HTS-Dreamer/.venv/bin/python
cd "$ROOT/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
"$PYTHON" -m pytest tests/test_h200_suite.py tests/test_prefix_persistence.py \
  tests/test_phase3h_semantics.py tests/test_pdyn_episode_mask.py tests/test_coarse_reconstruction_target.py -q
exec "$PYTHON" -u -m corewm_eval.h200_suite "$ROOT"
