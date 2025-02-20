#!/bin/bash
value=$(cat config/config_conda.txt)

echo "Sourcing conda.sh to enable 'conda activate'..."
source "$value"  # Adjust to your conda installation path

# Set the first conda environment path and activate it
current_dir=$(pwd)
SECOND_CONDA_ENV_PATH="$(pwd)/env_2"
conda activate "$SECOND_CONDA_ENV_PATH"
python main.py