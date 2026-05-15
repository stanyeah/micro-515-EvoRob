#!/bin/bash
#SBATCH --job-name=evorob_smoke
#SBATCH --account=micro-515
#SBATCH --partition=academic
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

export MUJOCO_GL=egl

module purge
module load gcc python py-virtualenv

cd ~/ER_course/micro-515-EvoRob
source .venv/bin/activate
pip install -e . --quiet

mkdir -p logs
python final_project_train.py --mode mind_only --smoke
