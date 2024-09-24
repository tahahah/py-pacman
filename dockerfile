# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8
RUN useradd -ms /bin/sh -u 1001 app
USER app

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY --chown=app:app . .
# ... existing code ...

# Switch to root to install packages and change permissions
USER root

# Ensure the directory exists and has the correct permissions
RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt/lists

# Install Xvfb, ALSA, and other necessary packages
RUN apt-get update && apt-get install -y \
    xvfb \
    alsa-utils \
    python3-pygame \
    && rm -rf /var/lib/apt/lists/*

# Add a script to start Xvfb and run your Python script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create necessary directories for X11
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Switch back to the app user
USER app

# Use the script as the entry point
ENTRYPOINT ["/start.sh"]