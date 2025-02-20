#!/bin/bash
#!/bin/bash
value=$(cat config/config_conda.txt)

echo "Sourcing conda.sh to enable 'conda activate'..."
source "$value"  # Adjust to your conda installation path

current_dir=$(pwd)
first_anaconda_env="$(pwd)/env_1"
second_anaconda_env="$(pwd)/env_2"

conda create -p $first_anaconda_env python=3.10 -y
conda create -p $second_anaconda_env python=3.10 -y
conda activate $first_anaconda_env
pip install khmer-nltk
pip install underthesea
pip install nltk
pip install numpy==1.25.0
pip install fastapi uvicorn pydantic python-multipart
conda activate $second_anaconda_env
cd OpenNMT-py
pip install -e .
cd ..
pip install fastapi uvicorn pydantic python-multipart
pip install nltk

