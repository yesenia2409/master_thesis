#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=70gb
#SBATCH --gres=gpu:A100:1

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

python3 -u ppo.py

