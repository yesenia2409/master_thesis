#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=07:00:00
#SBATCH --mem=20gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'

# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate thesis

# activate CUDA
module load devel/cuda/11.6

# call python script

python3 -u inference.py
# python3 -u try_GPT4_Api.py
