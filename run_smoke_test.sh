#!/bin/bash
#SBATCH --job-name=evorob_smoke
#SBATCH --account=micro-515
#SBATCH --partition=academic
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module purge
module load gcc python py-virtualenv

cd ~/ER_course/micro-515-EvoRob
source .venv/bin/activate
pip install -e . --quiet

mkdir -p logs
# Smoke run also exercises the Pool path (use 2 workers so we test parallelism).
python final_project_train.py \
    --mode mind_only \
    --smoke \
    --workers ${SLURM_CPUS_PER_TASK:-2}
