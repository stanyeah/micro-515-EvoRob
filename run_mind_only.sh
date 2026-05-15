#!/bin/bash
#SBATCH --job-name=evorob_mind_only
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mind_only_%j.out
#SBATCH --error=logs/mind_only_%j.err

set -e

module purge
module load gcc python py-virtualenv

cd ~/ER_course/micro-515-EvoRob
source .venv/bin/activate
pip install -e . --quiet

mkdir -p logs
python final_project_train.py --mode mind_only --seed ${SEED:-42}
