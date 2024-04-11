#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
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

# iterate over models
models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mixtral-8x7B-Instruct-v0.1" "mistralai/Mixtral-8x7B-v0.1")

for i in ${!models[*]}; do
    echo "model: ${models[$i]}"
    python3 -u model_selection.py \
      --model_name="${models[$i]}";
done
