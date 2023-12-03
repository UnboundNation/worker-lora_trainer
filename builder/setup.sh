#!/bin/bash

apt-get install ffmpeg libsm6 libxext6  -y

apt-get update && \
apt-get upgrade -y && \
apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && \
apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
    python3.10-dev python3-tk python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils build-essential && \
apt-get autoremove -y && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/*

# Clone kohya-ss
git clone https://github.com/bmaltais/kohya_ss.git && \
    cd kohya_ss && \
    git checkout 17f58c8bb9bc38773a870df58e1ae788abf6753b

# Cache models
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P /model_cache
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth -P /model_cache
