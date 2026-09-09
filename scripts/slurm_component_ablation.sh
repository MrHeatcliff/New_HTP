#!/usr/bin/env bash
#SBATCH --job-name=component-2x2
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/component-2x2-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/component-2x2-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
readonly ROOT="${1:?Supply frozen experiment directory}"
export HTP_RESEARCH_REPO=/home/vn-user0101/Dat/HTS-Dreamer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd "$ROOT/source"
readonly PYTHON="$HTP_RESEARCH_REPO/.venv/bin/python"
"$PYTHON" -m pytest tests/test_component_ablation.py tests/test_h200_suite.py tests/test_phase3h_semantics.py -q
exec "$PYTHON" -u -m corewm_eval.component_ablation "$ROOT"
