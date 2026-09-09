#!/usr/bin/env bash
#SBATCH --job-name=constraint-latest-audit
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-latest-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/constraint-latest-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly PYTHON="$REPO/.venv/bin/python"
readonly OUTPUT="$REPO/paper_artifacts/persistence_research/latest_${SLURM_JOB_ID}"
cd "$REPO"
mkdir -p "$OUTPUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
printf 'Job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
sha256sum dreamerv3/htp.py dreamerv3/agent_htp.py dreamerv3/configs.yaml corewm_eval/persistence*.py > "$OUTPUT/source_hashes.txt"
"$PYTHON" -m pytest tests/test_prefix_persistence.py tests/test_phase3h_semantics.py tests/test_pdyn_episode_mask.py -q
"$PYTHON" -m corewm_eval.persistence_long_audit --output-root "$OUTPUT/eval" \
  --run alien_whitened_p01 constraint_whitened_p01_alien_seed0
"$PYTHON" -m corewm_eval.persistence_representation current --source-run \
  production_runs/experiment_tracker/constraint_whitened_p01_alien_seed0 --output-root "$OUTPUT"
"$PYTHON" -m corewm_eval.persistence_latest_report "$OUTPUT"
printf 'Completed: %s\n' "$OUTPUT"
