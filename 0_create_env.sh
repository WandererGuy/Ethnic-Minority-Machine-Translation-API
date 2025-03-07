#!/bin/bash
value=$(cat config/config_conda.txt)
echo "Sourcing /home/username/miniconda3/etc/profile.d/conda.sh to enable 'conda activate'..."
source "$value"  # Adjust to your conda installation path
current_dir=$(pwd)
# first_anaconda_env="$(pwd)/env_1"
# echo "$first_anaconda_env"
second_anaconda_env="$(pwd)/env_2"
echo "$second_anaconda_env"
# conda create -p $first_anaconda_env python=3.10 -y
conda create -p $second_anaconda_env python=3.10 -y
conda activate $second_anaconda_env
# pip install khmer-nltk
# pip install underthesea
# pip install nltk
# pip install numpy==1.25.0
wget https://github.com/OpenNMT/OpenNMT-py/archive/refs/tags/2.3.0.tar.gz
tar -zxvf 2.3.0.tar.gz
mv OpenNMT-py-2.3.0 OpenNMT-py
cd OpenNMT-py
pip install -e .
cd ..
pip install nltk
pip install fastapi uvicorn pydantic python-multipart
