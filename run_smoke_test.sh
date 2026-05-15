#!/bin/bash
#SBATCH --job-name=evorob_smoke
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -e

module purge
module load gcc python py-virtualenv

cd ~/ER_course/micro-515-EvoRob
source .venv/bin/activate
pip install -e . --quiet

mkdir -p logs
python final_project_train.py --mode mind_only --smoke
