#!/bin/bash

# echo "Sourcing conda.sh to enable 'conda activate'..."
# source /home/manh264/miniconda3/etc/profile.d/conda.sh  # Adjust to your conda installation path
# # Set the second conda environment path and activate it
# current_dir=$(pwd)
# SECOND_CONDA_ENV_PATH="$(pwd)/env_2"

# echo "Activating second conda environment: $SECOND_CONDA_ENV_PATH"
# conda activate "$SECOND_CONDA_ENV_PATH"
# # n sample , fít 10000 in vocab 
onmt_build_vocab -config khmer-viet-no-bpe.yaml -n_sample 10000
# onmt_train -config khmer-viet-no-bpe.yaml -verbose -train_from models/run2/model_step_136000.pt
onmt_train -config khmer-viet-no-bpe.yaml -verbose --log_file output_train_log/output.log --tensorboard --tensorboard_log_dir output_tensorboard_log
