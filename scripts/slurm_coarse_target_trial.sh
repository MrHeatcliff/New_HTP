#!/usr/bin/env bash
#SBATCH --job-name=coarse-target-trial
#SBATCH --partition=gpu_junior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/coarse-target-%j.out
#SBATCH --error=/home/vn-user0101/Dat/HTS-Dreamer/slurm_logs/coarse-target-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
readonly REPO=/home/vn-user0101/Dat/HTS-Dreamer
readonly OUTPUT="$REPO/paper_artifacts/persistence_research/coarse_target_${SLURM_JOB_ID}"
cd "$REPO"
mkdir -p "$OUTPUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
printf 'Job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
sha256sum dreamerv3/htp.py dreamerv3/agent_htp.py dreamerv3/configs.yaml corewm_eval/coarse_target_trial.py > "$OUTPUT/source_hashes.txt"
"$REPO/.venv/bin/python" -m pytest tests/test_prefix_persistence.py tests/test_phase3h_semantics.py \
  tests/test_pdyn_episode_mask.py tests/test_coarse_reconstruction_target.py -q
exec "$REPO/.venv/bin/python" -m corewm_eval.coarse_target_trial "$OUTPUT"
