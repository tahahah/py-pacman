#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 &

# Set the DISPLAY environment variable
export DISPLAY=:99

# Start a virtual sound device
sudo modprobe snd-dummy

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


# Run your Python script
python dqn_pytorch.py -lay classic -e 5 -t -frs 4

# Keep the container running
tail -f /dev/null