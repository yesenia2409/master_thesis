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

lr=0.01
batch=2
epoch=15
# for i in ${!batch[*]}; do
#     echo "batch: ${batch[$i]}"
python3 -u training.py \
  --batch="${batch[$i]}" \
  --lr="${lr}" \
  --epochs="${epoch}"

# python3 -u inference.py
# done
