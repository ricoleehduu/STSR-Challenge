# Dockerfile

# Stage 1: Define Base Image and Environment
# Use official PyTorch image. The versions can be adjusted by participants.
ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Set environment variables for compilation (usually no need to change)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Fix potential GPG key errors during apt-get update in some environments
# (Usually no need to change)
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# [OPTIONAL] Uncomment the following line to use a mirror for faster package installation
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//https:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install essential system dependencies
RUN apt-get update && \
    apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Set up a non-root user for security
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /inputs /outputs && \
    chown -R algorithm:algorithm /opt/algorithm /inputs /outputs

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Stage 3: Install Python dependencies
# Update pip and install user-level packages
RUN python -m pip install --user -U pip

# --> ACTION REQUIRED: ADD YOUR DEPENDENCIES <--
# Copy and install all required packages from requirements.txt
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Stage 4: Copy your algorithm code and model weights
# --> ACTION REQUIRED: COPY YOUR FILES <--
# Copy your inference script, model definitions, and any utility files.
# Make sure the paths here match your project structure.
COPY --chown=algorithm:algorithm run_inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict.sh /opt/algorithm/
COPY --chown=algorithm:algorithm model/ /opt/algorithm/model/

# --- IMPORTANT ---
# You MUST copy your trained model weights into the Docker image.
# Replace 'your_model_weights.pth' with the actual filename.
COPY --chown=algorithm:algorithm your_model_weights.pth /opt/algorithm/your_model_weights.pth

# Stage 5: Set the entry point for the container
# The predict.sh script will be executed when the container runs.
CMD ["/bin/bash", "-c", "sh predict.sh"]