#!/usr/bin/env bash
# Logger remediation only; never a scientific paper run.
#SBATCH --job-name=corewm-logger-test
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/corewm-logger-%j.out
#SBATCH --error=slurm_logs/corewm-logger-%j.err

set -euo pipefail

readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly PYTHON=/home/vn-user0101/.conda/envs/htp/bin/python
: "${LOGGER_TEST_KIND:?LOGGER_TEST_KIND is required}"
: "${LOGGER_TEST_VARIANT:=Full}"

cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PYTHON" -m corewm_eval.logger_smoke run \
  --kind "$LOGGER_TEST_KIND" \
  --variant "$LOGGER_TEST_VARIANT"
