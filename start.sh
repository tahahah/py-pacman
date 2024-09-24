#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 &

# Set the DISPLAY environment variable
export DISPLAY=:99

# Start a virtual sound device
sudo modprobe snd_dummy

# Run your Python script
python dqn_pytorch.py -lay classic -e 10 -t -frs 4

# Keep the container running
tail -f /dev/null