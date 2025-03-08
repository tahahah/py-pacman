import gymnasium as gym
import numpy as np
import pygame as pg
import time

# Import the new environment
from src.env.pacman_env_new import PacmanEnv

def test_gymnasium_env_no_render():
    """
    Test the Gymnasium-compatible Pacman environment without rendering
    """
    print("Testing Gymnasium-compatible Pacman environment without rendering...")
    
    # Initialize Pygame
    pg.init()
    
    # Create the environment
    env = PacmanEnv(
        layout='classic',
        render_mode='rgb_array',
        enable_render=False,
        state_active=False,
        player_lives=3
    )
    
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(f"Observation shape: {observation.shape}")
    print(f"Info: {info.keys()}")
    
    # Take some random actions
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1} - Action: {action}, Reward: {reward}")
        
        if terminated or truncated:
            print("Episode finished")
            observation, info = env.reset()
    
    # Close the environment
    env.close()
    pg.quit()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_gymnasium_env_no_render()
