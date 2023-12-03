# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

LABEL org.opencontainers.image.source=https://github.com/UnboundNation/worker-lora_trainer



# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive



# Set the working directory
WORKDIR /

# Install system packages, clone repo, and cache models
COPY builder/setup.sh /setup.sh
RUN bash /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade torch>=2.0.1+cu118 torchvision>=0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip install --upgrade setuptools && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src /kohya_ss



CMD python3 -u handler.py
