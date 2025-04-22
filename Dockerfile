# # Use CUDA 11.3.1 on Ubuntu 20.04 as the build/runtime base
# FROM ubuntu:22.04

# # Step 2: Set the working directory in the container
# WORKDIR /app
# # COPY . /app

# # Step 3: Install any required dependencies
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     curl \
#     git \
#     # Add any additional dependencies your application requires
#     && rm -rf /var/lib/apt/lists/*
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# 1) Make all apt installs non‑interactive
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# 2) Install everything you need in one go, including tzdata
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      curl \
      git \
      cmake \
      build-essential \
      pkg-config \
      libgoogle-perftools-dev \
      tzdata \
 && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/WandererGuy/Ethnic-Minority-Machine-Translation-API.git
WORKDIR /app/Ethnic-Minority-Machine-Translation-API
RUN apt-get update
RUN apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev
RUN git clone https://github.com/google/sentencepiece.git
WORKDIR /app/Ethnic-Minority-Machine-Translation-API/sentencepiece
RUN mkdir build
WORKDIR /app/Ethnic-Minority-Machine-Translation-API/sentencepiece/build
RUN cmake ..
RUN make -j $(nproc)
RUN make install
RUN ldconfig -v
WORKDIR /app/Ethnic-Minority-Machine-Translation-API
RUN tar -zxvf 2.3.0.tar.gz
RUN mv OpenNMT-py-2.3.0 OpenNMT-py
WORKDIR /app/Ethnic-Minority-Machine-Translation-API/OpenNMT-py
RUN pip install -e .
WORKDIR /app/Ethnic-Minority-Machine-Translation-API
RUN pip install nltk
RUN pip install fastapi uvicorn pydantic python-multipart
RUN pip install transformers
RUN pip install OpenNMT-tf

RUN pip install tensorflow
RUN pip install 'keras<3.0.0'
# RUN pip install 'keras<3.0.0' mediapipe-model-maker

# RUN pip install "pyyaml>6.0.0" "keras<3.0.0" "tensorflow<2.16" "tf-models-official<2.16" 
RUN pip install mediapipe-model-maker --no-deps
RUN mkdir data

# Make the script executable
# RUN chmod +x /app/script.sh
RUN ln -s /usr/bin/python3 /usr/bin/python
# port in config/config.ini
RUN pip install gdown
# RUN gdown --folder https://drive.google.com/drive/folders/13i46pilo1kOMIAn-t2XKXnvCEDYuvOb9
RUN pip install pandas
RUN mkdir target_source
WORKDIR /app/Ethnic-Minority-Machine-Translation-API/target_source
RUN gdown https://drive.google.com/uc?id=1qvLCV9xzMOZvlsvqRvBjksZmNo_LFC6C
WORKDIR /app/Ethnic-Minority-Machine-Translation-API
EXPOSE 5021 

# docker build  --no-cache -t nmt_main .    
# docker run -it --gpus all -p 5021:5021 nmt_main

# # Update the package list and install dependencies
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     curl \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update \
#     && apt-get install -y python3.10 python3.10-venv python3.10-dev


# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# docker run -it --gpus all -e CUDA_LAUNCH_BLOCKING=1 -p 5021:5021 -v D:\MANH_T04:/app/Ethnic-Minority-Machine-Translation-API/checkpoints nmt_main
# RUN gdown https://drive.google.com/uc?id=1qvLCV9xzMOZvlsvqRvBjksZmNo_LFC6C
