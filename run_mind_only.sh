#!/bin/bash
#SBATCH --job-name=evorob_mind_only
#SBATCH --account=micro-515
#SBATCH --partition=academic
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --output=logs/mind_only_%j.out
#SBATCH --error=logs/mind_only_%j.err

export MUJOCO_GL=egl

# Cap intra-process threading so each Pool worker uses exactly one thread.
# Without this, every worker would try to use all $SLURM_CPUS_PER_TASK cores
# for numpy/BLAS/MuJoCo, oversubscribing the node by ~64x.
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
python final_project_train.py \
    --mode mind_only \
    --seed ${SEED:-42} \
    --workers ${SLURM_CPUS_PER_TASK:-1}
