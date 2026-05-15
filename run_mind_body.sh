#!/bin/bash
#SBATCH --job-name=evorob_mind_body
#SBATCH --account=micro-515
#SBATCH --partition=academic
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --output=logs/mind_body_%j.out
#SBATCH --error=logs/mind_body_%j.err

export MUJOCO_GL=egl

module purge
module load gcc python py-virtualenv

cd ~/ER_course/micro-515-EvoRob
source .venv/bin/activate
pip install -e . --quiet

mkdir -p logs
python final_project_train.py --mode mind_body --seed ${SEED:-42}
