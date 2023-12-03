#!/bin/bash

virtualenv --python="/opt/homebrew/bin/python3.10" .venv && source .venv/bin/activate

pip install --upgrade pip && \
    pip install --upgrade torch>=2.0.1+cu118 torchvision>=0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --upgrade setuptools && \
    pip install --upgrade -r builder/requirements.txt && \


# Clone kohya-ss
git clone https://github.com/bmaltais/kohya_ss.git && \
    cd kohya_ss && \
    git checkout 17f58c8bb9bc38773a870df58e1ae788abf6753b

# Cache models
mkdir model_cache

#Only if not present already
#wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P model_cache
#wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth -P model_cache
