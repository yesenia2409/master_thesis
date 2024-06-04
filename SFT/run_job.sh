#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
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

# lr=0.00002
# batch=2
# epoch=1
# for i in ${!batch[*]}; do
#    echo "batch: ${batch[$i]}"
# python3 -u training.py \
#   --batch="${batch[$i]}" \
#   --lr="${lr}" \
#   --epochs="${epoch}"
# done

python3 -u inference.py
