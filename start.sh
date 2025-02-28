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
XVFB_PID=$!

# Wait for Xvfb to start
sleep 2

# Check if Xvfb started successfully
if ! ps -p $XVFB_PID > /dev/null; then
    echo "Error: Xvfb failed to start"
    exit 1
fi

# Set the DISPLAY environment variable
export DISPLAY=:99

# Try to start a virtual sound device, but continue if it fails
if ! modprobe snd-dummy 2>/dev/null; then
    echo "Warning: Could not load snd-dummy module. Audio may not work."
fi

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

# Run your Python script with unbuffered output and redirect output to both console and log file
echo "Starting Pac-Man Reinforcement Learning with Rainbow DQN..."
python -u dqn_pytorch.py -lay classic -e 30000001 -t -frs 4 | tee /app/training.log

# If the Python script exits, clean up Xvfb
if ps -p $XVFB_PID > /dev/null; then
    echo "Shutting down Xvfb..."
    kill $XVFB_PID
fi

# Keep the container running only if explicitly requested
if [ "$1" = "keep-alive" ]; then
    echo "Keeping container alive as requested..."
    tail -f /dev/null
fi