# Use the official CUDA 12.2 base image from NVIDIA
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install Python 3.8 from deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3.8-distutils \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip


# Copy the uv tool from the GitHub container registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

# Add a script to start Xvfb and run your Python script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt and error if pip errors
RUN python3.8 -m pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY --chown=app:app . .

# Ensure the directory exists and has the correct permissions
RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt/lists

# Install Xvfb, ALSA, and other necessary packages
RUN apt-get update && apt-get install -y \
    xvfb \
    alsa-utils \
    python3-pygame \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories for X11
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Use the script as the entry point
ENTRYPOINT ["/start.sh"]