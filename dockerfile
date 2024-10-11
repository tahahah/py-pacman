# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8
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

# Use the script as the entry point
ENTRYPOINT ["/start.sh"]