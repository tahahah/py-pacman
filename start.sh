#!/bin/bash

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
fi

# Run your Python script with unbuffer to see logs in real time
unbuffer python dqn_pytorch.py -lay classic -e 10000 -t -frs 1

# Keep the container running
tail -f /dev/null