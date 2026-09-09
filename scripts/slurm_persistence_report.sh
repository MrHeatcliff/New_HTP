#!/usr/bin/env bash
# Submit only after the preceding research allocation has finished.
#SBATCH --job-name=constraint-analysis
#SBATCH --partition=cpu_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-analysis-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-analysis-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via sbatch}"
readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly ROOT=${1:?usage: sbatch scripts/slurm_persistence_report.sh RESEARCH_OUTPUT}
cd "$REPO"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
printf 'Statistical analysis job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
exec "$REPO/.venv/bin/python" -m corewm_eval.persistence_research_report "$ROOT"
