#!/usr/bin/env bash
# Submit one Slurm job per EA seed (separate nodes / separate result dirs).
#
# Usage (on the cluster, from the repo root):
#   chmod +x submit_seed_jobs.sh run_mind_only.sh
#   ./submit_seed_jobs.sh mind_only 0 1
#   ./submit_seed_jobs.sh mind_body 2 3 4
#
# Requires: run_mind_only.sh or run_mind_body.sh in the same directory,
#           logs/ will be created by the Slurm scripts.
#
# Each job sets SEED for final_project_train.py → results/final_<mode>/seed_<n>/

set -euo pipefail

MODE="${1:?usage: $0 mind_only|mind_body SEED1 [SEED2 ...]}"
shift
if [[ $# -lt 1 ]]; then
  echo "usage: $0 mind_only|mind_body SEED1 [SEED2 ...]" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT="${SCRIPT_DIR}/run_${MODE}.sh"
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "error: missing ${RUN_SCRIPT}" >&2
  exit 1
fi

for seed in "$@"; do
  echo "Submitting ${MODE}  seed=${seed}  →  $(basename "$RUN_SCRIPT")"
  # ALL keeps your login environment; SEED overrides for this job.
  sbatch --export=ALL,SEED="${seed}" "$RUN_SCRIPT"
done
