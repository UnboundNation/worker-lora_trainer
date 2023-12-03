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
    python3.10 pip install --upgrade pip && \
    python3.10 pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    python3.10 pip install --upgrade setuptools && \
    python3.10 pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD /src /src


CMD python3.10 -u handler.py
