#!/usr/bin/env bash
#SBATCH --job-name=constraint-figures
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-figures-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-figures-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
cd /home/vn-user0101/Dat/HTS-Dreamer
readonly OUTPUT="paper_artifacts/full_vs_dreamerv3_constraint_suite_learning_curves"
test ! -e "$OUTPUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/corewm-matplotlib
readonly PYTHON=/home/vn-user0101/.conda/envs/htp/bin/python
"$PYTHON" -m pytest tests/test_constraint_suite_curves.py -q
exec "$PYTHON" -u -m corewm_eval.constraint_suite_curves \
  --suite-root production_runs/constraint_full26_seed0_OzirkNB8 \
  --existing-figures paper_artifacts/full_vs_dreamerv3_learning_curves \
  --output-dir "$OUTPUT"
