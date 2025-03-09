"""
Visualize the exact image input that's being fed into the neural network.
This script will:
1. Create the same environment setup as in training
2. Process a single observation through the same preprocessing pipeline
3. Display the original and processed images side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import gymnasium as gym
from src.env.pacman_env_new import PacmanEnv

def main():
    # Create the environment with render_mode explicitly set to rgb_array
    env = PacmanEnv(layout="classic", enable_render=True, render_mode="rgb_array", state_active=True, player_lives=3)
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Run several steps to ensure the game is properly initialized
    for i in range(20):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Explicitly get the RGB array from the environment
    original_obs = env.render()
    
    if original_obs is None or original_obs.size == 0:
        print("Error: Could not get a valid observation from the environment.")
        print("Trying alternative method to get observation...")
        # Try to get the observation directly from the game's screen
        original_obs = env.get_screen_rgb_array()
    
    print(f"Original observation shape: {original_obs.shape}")
    
    # Process the observation the same way as in training
    # Resize to 84x84
    processed_obs = cv2.resize(original_obs, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Convert to channel-first format (C, H, W) for the neural network
    processed_obs_chw = np.transpose(processed_obs, (2, 0, 1))
    print(f"Processed observation shape (CHW): {processed_obs_chw.shape}")
    
    # Create a figure to display the images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original observation
    ax1.imshow(original_obs)
    ax1.set_title(f"Original Observation\nShape: {original_obs.shape}")
    ax1.axis('off')
    
    # Display processed observation (in HWC format for visualization)
    ax2.imshow(processed_obs)
    ax2.set_title(f"Network Input\nShape: {processed_obs_chw.shape}")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('network_input_visualization.png')
    
    # Additional visualization: Show each channel separately
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display each channel
    for i, ax in enumerate(axes):
        ax.imshow(processed_obs_chw[i], cmap='gray')
        ax.set_title(f"Channel {i}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('network_input_channels.png')
    
    # Print pixel values to verify we have actual data
    print("\nSample of original observation pixel values:")
    print(original_obs[0:5, 0:5, 0])  # First 5x5 pixels of first channel
    
    print("\nSample of processed observation pixel values:")
    print(processed_obs_chw[0, 0:5, 0:5])  # First 5x5 pixels of first channel
    
    # Close the environment
    env.close()
    
    print("\nImages saved as 'network_input_visualization.png' and 'network_input_channels.png'")
    print("The neural network receives the processed image in the format: (3, 84, 84)")
    print("This is a 3-channel image (RGB) with dimensions 84x84 pixels")

if __name__ == "__main__":
    main()
