# Use the official CUDA 12.2 base image from NVIDIA
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Sydney

# Install Miniconda and required packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && /opt/conda/bin/conda clean -a

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH


# Install Python 3.8 and PyTorch nightly build with CUDA 12.2
RUN conda install python=3.8.2 -y \
    && conda install -c conda-forge libjpeg-turbo libpng -y \
    && conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y


#
RUN cd /opt/conda/lib \
    && mkdir backup \
    && mv libstd* backup \
    && cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./ \
    && ln -s libstdc++.so.6 libstdc++.so \
    && ln -s libstdc++.so.6 libstdc++.so.6.0.19

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

# Ensure pip is below version 24.1
RUN pip install "pip<24.1" wheel==0.36.2 setuptools==56.0.0

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

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
# CMD ["python", "-u", "dqn_pytorch.py", "-lay", "classic", "-e", "20001", "-t", "-frs", "4"]
ENTRYPOINT [ "/start.sh" ]