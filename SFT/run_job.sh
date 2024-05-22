#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=80gb
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

lrs=(0.01)
for i in ${!lrs[*]}; do
    echo "learning rate: ${lrs[$i]}"
    python3 -u training.py \
    --lr="${lrs[$i]}";

# python3 -u inference.py
done
