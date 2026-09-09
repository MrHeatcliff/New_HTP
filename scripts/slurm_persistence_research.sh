#!/usr/bin/env bash
# One allocation; all research stages execute sequentially on the compute node.
#SBATCH --job-name=constraint-research
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-research-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-research-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Run this script via sbatch}"
: "${SLURM_JOB_NODELIST:?Missing Slurm compute allocation}"
readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly PYTHON="$REPO/.venv/bin/python"
readonly OUTPUT="$REPO/paper_artifacts/persistence_research/slurm_${SLURM_JOB_ID}"
cd "$REPO"
mkdir -p "$OUTPUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
printf 'Job %s on %s; sequential research stages\n' "$SLURM_JOB_ID" "$(hostname)"
git diff -- dreamerv3 corewm_eval tests scripts > "$OUTPUT/source_changes.patch"
sha256sum dreamerv3/htp.py dreamerv3/agent_htp.py dreamerv3/configs.yaml \
  corewm_eval/persistence*.py > "$OUTPUT/source_hashes.txt"

"$PYTHON" -m pytest tests/test_prefix_persistence.py tests/test_phase3h_semantics.py -q
"$PYTHON" -m corewm_eval.persistence_long_audit --output-root "$OUTPUT/long_audit"
"$PYTHON" -m corewm_eval.persistence_experiment --whitening --steps 5000 \
  --output-root "$OUTPUT/synthetic"
"$PYTHON" -m corewm_eval.persistence_whitening_trial --output-root "$OUTPUT/whitening_trial"
printf 'All research stages completed: %s\n' "$OUTPUT"
