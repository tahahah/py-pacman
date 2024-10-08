#!/bin/bash


# Check if Xvfb is running and shut it down if it is
if pgrep Xvfb > /dev/null; then
    echo "Xvfb is already running. Shutting it down..."
    pkill Xvfb
    echo "Xvfb has been shut down."
    if [ -f /tmp/.X99-lock ]; then
        echo "Removing /tmp/.X99-lock..."
        rm /tmp/.X99-lock
        echo "/tmp/.X99-lock has been removed."
    fi
    # Wait until Xvfb process is completely terminated
    while pgrep Xvfb > /dev/null; do
        echo "Waiting for Xvfb to terminate..."
        sleep 1
    done
else
    echo "Xvfb is not running."
fi

# Ensure the lock file is removed before starting a new Xvfb instance
if [ -f /tmp/.X99-lock ]; then
    echo "Removing /tmp/.X99-lock..."
    rm /tmp/.X99-lock
    echo "/tmp/.X99-lock has been removed."
fi

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 &

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 &

# Set the DISPLAY environment variable
export DISPLAY=:99

# Start a virtual sound device
modprobe snd-dummy

# Configure ALSA to use the dummy sound card
cat <<EOL > ~/.asoundrc
pcm.!default {
    type hw
    card 0
}
ctl.!default {
    type hw
    card 0
}
EOL

# Install expect if not already installed
if ! command -v unbuffer &> /dev/null
then
    apt-get update
    apt-get install -y expect
    ln -s /usr/bin/unbuffer /usr/local/bin/unbuffer  # Ensure unbuffer is in PATH
fi

# Run your Python script with unbuffer to see logs in real time
unbuffer python dqn_pytorch.py -lay classic -e 10000 -t -frs 4 -rmq

# Keep the container running
tail -f /dev/null