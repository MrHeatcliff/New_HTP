#!/usr/bin/env bash
#SBATCH --job-name=corewm-paper-eval
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/manuscript-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/manuscript-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
readonly RUNROOT="${1:?Frozen run directory}"
readonly PYTHON=/home/vn-user0101/Dat/HTS-Dreamer/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$RUNROOT/source"
"$PYTHON" -m pip install --no-index --find-links "$RUNROOT/wheels" --target "$RUNROOT/dependencies" \
  -r corewm_eval/requirements-manuscript.txt
export PYTHONPATH="$RUNROOT/dependencies${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m pip freeze --path "$RUNROOT/dependencies" > "$RUNROOT/dependencies.txt"
"$PYTHON" -m pytest tests/test_manuscript_eval.py tests/test_corewm_eval_math.py -q
exec "$PYTHON" -u -m corewm_eval.manuscript_suite "$RUNROOT/results"
