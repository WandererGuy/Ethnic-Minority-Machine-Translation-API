#!/bin/bash
value=$(cat config/config_conda.txt)

echo "Sourcing conda.sh to enable 'conda activate'..."
source "$value"  # Adjust to your conda installation path

# Set the first conda environment path and activate it
current_dir=$(pwd)
FIRST_CONDA_ENV_PATH="$(pwd)/env_1"
SECOND_CONDA_ENV_PATH="$(pwd)/env_2"
echo "Activating first conda environment: $FIRST_CONDA_ENV_PATH"
conda activate "$FIRST_CONDA_ENV_PATH"

# Run your Python scripts with delays
echo "Running check_prepare.py..."
python 1_check_prepare.py

echo "Sleeping for 2 seconds..."
sleep 2

echo "Running split.py..."
python 2_split.py

echo "Sleeping for 4 seconds..."
sleep 4

echo "Running tokenize_source.py..."
python 3_tokenize_source.py

echo "Sleeping for 4 seconds..."
sleep 4

echo "Running tokenize_target.py..."
python 4_tokenize_target.py

echo "Sleeping for 4 seconds..."
sleep 4

echo "Running config.py..."
python 5_config.py

echo "Sleeping for 4 seconds..."
sleep 4

echo "Activating second conda environment: $SECOND_CONDA_ENV_PATH"
conda activate "$SECOND_CONDA_ENV_PATH"

echo "Starting train.sh..."
bash 6_train-no-bpe.sh

echo "Sleeping for 4 seconds..."
sleep 4
python 8_create_sample_translate.py

echo "Sleeping for 4 seconds..."
sleep 4
bash 9_translate-no-bpe.sh

echo "Sleeping for 4 seconds..."
sleep 4
python 9_1_refine_translate.py