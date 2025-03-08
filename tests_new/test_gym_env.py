import gymnasium as gym
import numpy as np
import pygame as pg
import time

# Import the new environment
from src.env.pacman_env_new import PacmanEnv

def test_gymnasium_env():
    """
    Test the Gymnasium-compatible Pacman environment
    """
    print("Testing Gymnasium-compatible Pacman environment...")
    
    # Initialize Pygame
    # pg.init()
    
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
    for _ in range(100):
        action = int(env.action_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"Action: {action}, Reward: {reward}")
        
        # Process pygame events to keep the window responsive
        for event in pg.event.get():
            if event.type == pg.QUIT:
                env.close()
                pg.quit()
                return
        
        # Small delay to make the game visible
        time.sleep(0.1)
        
        if terminated or truncated:
            print("Episode finished")
            observation, info = env.reset()
    
    # Close the environment
    env.close()
    pg.quit()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_gymnasium_env()
