from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import pygame as pg
from gym import spaces
from gym.utils import seeding

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
        seed

    Its extends the gym.Env class
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}
    reward_range = (-10, 5)

    def __init__(self, layout: str, enable_render=True, state_active=False, player_lives: int = 3):
        """
        PacmanEnv constructor

        :param layout: the layout of the game
        :param frame_to_skip: the frame to skip during training
        :param enable_render: enabling the display of the game screen
        :param state_active: enabling the display of the state matrix
        """
        self.layout = layout
        self.state_active = state_active
        self.enable_render = enable_render
        if enable_render:
            pg.init()
        self.action_space = spaces.Discrete(Action.__len__())
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(
            maze=self.maze,
            screen=Controller.get_screen(state_active, self.width, self.height) if enable_render else None,
            sounds_active=False,
            state_active=state_active,
            agent=None
        )
        self.timer = 0
        self.reinit_game = False
        self.player_lives = player_lives

        self.observation_space = spaces.Space(shape=self.get_screen_rgb_array().shape, dtype=int)

        self.seed()

    # def __del__(self):
    #     if self.enable_render:
    #         pg.quit()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).

        :param seed:
        :return:
        """
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, mode='human'):
        """
        Reset and Restart the Pacman Environment
        """
        self.game.maze.reinit_map()
        self.game.restart()
        self.game.player.regenerate()
        self.game.score = 0
        self.game.mode_timer = 0
        self.game.ghosts_timer = 0
        self.game.set_mode(GameMode.normal)
        self.game.make_ghosts_normal()
        self.game.player.lives = self.player_lives
        if mode == 'human':
            return self.get_state_matrix()
        elif mode == 'rgb_array':
            return self.get_screen_rgb_array()
        elif mode == 'info':
            return self.get_info_dict()

    def render(self, mode='human'):
        """
        Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        :param mode: the mode to render
        :return: the rba_array if the anonymous mode is active
        """
        if mode == 'human':
            if self.enable_render:
                self.game.init_screen()
                self.game.draw()
                pg.display.flip()
        elif mode == 'rgb_array':
            return self.get_screen_rgb_array()

    def close(self):
        self.__del__()

    def step(self, action: Union[Action, int]):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: action to perform in the environment
        :return: a tuple containing the following: observation, reward, done, info
        """
        action = Action(int(action)) if type(action) is int else action
        rewards = self._one_step_action(action)
        done = self.get_mode() == GameMode.game_over or self.get_mode() == GameMode.black_screen
        obs = self.get_screen_rgb_array()
        info = self.get_info_dict()
        return obs, rewards, done, info

    def get_info_dict(self) -> Dict[str, Any]:
        ghosts_pixel_pos = [ghost.get_pixel_position() for ghost in self.game.ghosts]
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
            'player action': self.game.player.current_action
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

        self.game.player.change_player_vel(action, self.game)
        self.game.move_players()

        succ_reward = self.game.total_rewards

        return succ_reward - prev_reward

    def get_mode(self) -> GameMode:
        """

        :return: the current mode of the Game, is an instance of the Enum GameMode
        """
        return self.game.game_mode

    def get_state_matrix(self) -> np.ndarray:
        return self.maze.state_matrix

    def get_player_position(self) -> Tuple[int, int]:
        """

        :return: a tuple containing respectively the x and y coordinate
                int the grid
        """
        return self.game.player.nearest_col, self.game.player.nearest_row

    def get_player_pixel_position(self) -> Tuple[int, int]:
        """

        :return: a tuple containing respectively the x and y pixel position in the
                game
        """
        return self.game.player.x, self.game.player.y

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

    def get_screen_rgb_array(self):
        screen = self.game.screen.copy()
        return pg.surfarray.pixels3d(screen)
