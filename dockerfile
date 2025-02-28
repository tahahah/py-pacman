# Use the official CUDA 12.1 base image from NVIDIA
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney
ENV PATH="/root/.local/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    xvfb \
    alsa-utils \
    python3-pygame \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for X11
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Ensure the directory exists and has the correct permissions
RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt/lists

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh


# Set the working directory in the container
WORKDIR /app

RUN uv venv --python=3.8.2 /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv
# Place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first
COPY requirements.txt .

# Install PyTorch with CUDA support and other dependencies
# Using a more direct approach to ensure proper installation
RUN uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN uv pip install "pip<24.1" wheel==0.36.2 setuptools==56.0.0
RUN uv pip install -r requirements.txt

# Verify Python and PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Add a script to start Xvfb and run your Python script
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]