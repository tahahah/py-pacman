from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import pygame as pg
import gymnasium as gym
from gymnasium import spaces

from src.controller import Controller
from src.game import Game
from src.map import Map
from src.utils.action import Action
from src.utils.game_mode import GameMode


class PacmanEnv(gym.Env):
    """
    Reinforcement Learning Environment wrapper for the game.
    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    An environment can be partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close

    Its extends the gymnasium.Env class
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    reward_range = (-10, 5)

    def __init__(self, layout: str, render_mode: str = 'human', enable_render=True, state_active=False, player_lives: int = 3):
        """
        Initialize the Pacman environment.

        Args:
            layout: Path to the layout file or predefined layout name.
            render_mode: The render mode ('human', 'rgb_array').
            enable_render: Whether to enable rendering.
            state_active: Whether to display the state matrix.
            player_lives: Number of lives the player has.
        """
        # super().__init__()

        # Initialize Pygame if we're using any rendering
        # if enable_render:
        #     pg.init()

        self.render_mode = render_mode
        self.layout = layout
        self.state_active = state_active
        self.enable_render = enable_render
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(Action.__len__())
        
        # Create map and game objects
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(
            maze=self.maze,
            screen=Controller.get_screen(state_active, self.width, self.height) if self.enable_render else None,
            sounds_active=False,
            state_active=state_active,
            agent=None
        )
        
        # Initialize game_mode explicitly to avoid the AttributeError
        self.game.game_mode = GameMode.normal
        
        # Get screen shape for observation space
        screen_shape = self.get_screen_rgb_array().shape
        self.observation_space = spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8)
        
        # Initialize other attributes
        self.timer = 0
        self.reinit_game = False
        self.player_lives = player_lives

        # For automatic reset
        self._np_random = None

    def _get_obs(self):
        """
        Get the current observation
        
        :return: the current observation
        """
        return self.get_screen_rgb_array()
    
    def _get_info(self):
        """
        Get the current info dictionary
        
        :return: the current info dictionary
        """
        return self.get_info_dict()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.

        Args:
            seed: The seed for the random number generator.
            options: Additional options for resetting the environment.

        Returns:
            observation: The initial observation.
            info: Additional information.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the game
        self.game.maze.reinit_map()
        self.game.restart()
        self.game.player.regenerate()
        self.game.score = 0
        self.game.mode_timer = 0
        self.game.ghosts_timer = 0
        self.game.set_mode(GameMode.normal)
        self.game.make_ghosts_normal()
        self.game.player.lives = self.player_lives
        
        # Reset timer and game state
        self.timer = 0
        self.reinit_game = False
        
        # Ensure the game is drawn before capturing the observation
        if self.enable_render and self.game.screen is not None:
            # Clear the screen before drawing to prevent trail effect
            self.game.screen.fill((0, 0, 0))
            self.game.draw()
            pg.display.flip()
        
        # Return observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def render(self):
        """
        Render the current state of the environment.

        Returns:
            For 'human' render mode, returns None.
            For 'rgb_array' render mode, returns an RGB array of the screen.
        """
        if self.render_mode is None:
            return None
            
        if not self.enable_render:
            return None
            
        # Initialize the screen if needed
        if self.game.screen is None and self.enable_render:
            self.game.screen = Controller.get_screen(self.state_active, self.width, self.height)
            
        # Draw the game
        self.game.init_screen()
        self.game.draw()
        
        # Return based on render mode
        if self.render_mode == 'human':
            pg.display.flip()
            return None
        elif self.render_mode == 'rgb_array':
            return self.get_screen_rgb_array()

    def close(self):
        """
        Close the environment
        """
        if self.enable_render:
            pg.quit()

    def step(self, action: Union[Action, int]):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).

        :param action: action to perform in the environment
        :return: a tuple containing the following: observation, reward, terminated, truncated, info
        """
        action = Action(int(action)) if type(action) is int or np.int64 else action
        reward = self._one_step_action(action)
        
        # Check if the episode is terminated
        terminated = self.get_mode() == GameMode.game_over or self.get_mode() == GameMode.black_screen
        
        # In this environment, we don't have truncation (early stopping)
        truncated = False
        
        # Ensure the game is drawn before capturing the observation
        if self.enable_render and self.game.screen is not None:
            # Clear the screen before drawing to prevent trail effect
            self.game.screen.fill((0, 0, 0))
            self.game.draw()
            pg.display.flip()
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def get_info_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary of information about the current state of the environment.

        Returns:
            A dictionary containing information about the current state.
        """
        ghosts_pixel_pos = [(ghost.x, ghost.y) for ghost in self.game.ghosts]
        number_of_scared_ghosts = sum([ghost.is_vulnerable() for ghost in self.game.ghosts])
        info = {
            'win': self.get_mode() == GameMode.black_screen,
            'player position': self.get_player_position(),
            'player pixel position': self.get_player_pixel_position(),
            'player lives': self.game.player.lives,
            'game mode': self.get_mode().value,
            'game score': self.game.score,
            'number of scared ghosts': number_of_scared_ghosts,
            'state matrix': self.get_state_matrix(),
            'ghosts_pixel_pos': ghosts_pixel_pos,
            'player vel': self.game.player.get_vel(),
            'player action': self.game.player.current_action,
            'pellets left': self.maze.get_number_of_pellets()
        }
        return info

    def _one_step_action(self, action: Union[Action, int]) -> int:
        """
        Performs only one step of the given action in the environment

        :param action: action to perform
        :return: the reward obtained after performing the action
        """
        self.check_game_mode()

        if self.get_mode() is GameMode.game_over:
            return 0
        elif self.get_mode() is GameMode.black_screen:
            return 0
        elif self.reinit_game:
            self.reinit_game = False
            return 0

        prev_reward = self.game.total_rewards
        prev_pellets = self.game.maze.get_number_of_pellets()

        self.game.player.change_player_vel(action, self.game)
        self.game.move_players()

        succ_reward = self.game.total_rewards
        current_pellets = self.game.maze.get_number_of_pellets()
        reward = succ_reward - prev_reward
        reward += 0.1 * (prev_pellets - current_pellets)
        reward -= 0.02
        pacman_x, pacman_y = self.game.player.x, self.game.player.y
        ghost_dist = min(ghost.distance_to_pacman(pacman_x, pacman_y) for ghost in self.game.ghosts)
        if ghost_dist < 5:
            reward -= 0.5

        return reward

    def get_mode(self) -> GameMode:
        """
        Get the current game mode

        :return: the current game mode
        """
        return self.game.game_mode

    def check_game_mode(self):
        mode = self.get_mode()

        if self.maze.get_number_of_pellets() == 0:
            self.game.set_mode(GameMode.black_screen)
            return

        if mode is GameMode.hit_ghost:
            self.game.player.lives -= 1
            if self.game.player.lives == 0:
                self.game.set_mode(GameMode.game_over)
            else:
                self.game.init_players_in_map()
                self.game.make_ghosts_normal()
                self.game.set_mode(GameMode.normal)
                self.reinit_game = True
        elif mode == GameMode.wait_after_eating_ghost:

            self.game.move_ghosts()

            if self.maze.get_number_of_pellets() == 0:
                self.game.set_mode(GameMode.black_screen)
            elif self.game.are_all_ghosts_vulnerable():
                self.game.set_mode(GameMode.change_ghosts)
            elif self.game.are_all_ghosts_normal():
                self.game.set_mode(GameMode.normal)

        self.game.check_ghosts_state()

    def get_state_matrix(self) -> np.ndarray:
        """
        Get the state matrix of the maze.

        Returns:
            The state matrix as a numpy array.
        """
        return self.maze.state_matrix

    def get_screen_rgb_array(self):
        screen = self.game.screen.copy()
        return pg.surfarray.pixels3d(screen)

    def get_player_position(self):
        """
        Get the player position in the grid

        :return: the player position in the grid
        """
        return self.game.player.get_position()

    def get_player_pixel_position(self) -> Tuple[int, int]:
        """
        Get the player's pixel position in the game.

        Returns:
            A tuple containing the x and y pixel position of the player.
        """
        return self.game.player.get_pixel_pos()
