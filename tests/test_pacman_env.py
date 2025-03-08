import logging

import numpy as np

from src.env.pacman_env_new import PacmanEnv
from src.utils.action import Action
from src.utils.game_mode import GameMode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def test_action_space():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=False)
    assert type(env.action_space.sample()) is int


def test_env_reset():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=False
    )
    env.reset()
    assert env.get_mode() == GameMode.normal
    assert env.game.player.anim_frame == 3
    assert env.game.player.vel_x == 0
    assert env.game.player.vel_y == 0


def test_env_render():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=True
    )
    env.reset()
    env.render()
    assert 1 == 1


def test_env_without_render():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=5,
        enable_render=False
    )
    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        assert type(action) is int
        assert type(obs) is np.ndarray
        assert type(rewards) is int
        assert type(done) is bool
        assert type(info) is dict
        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            break
    env.close()


def test_env_with_render():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=True
    )
    env.reset()
    for i in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        assert type(action) is int
        assert type(obs) is np.ndarray
        assert type(rewards) is int
        assert type(done) is bool
        assert type(info) is dict
        logger.info("reawrds {}".format(rewards))
        if done:
            logger.info("Episode finished after {} timesteps".format(i + 1))
            break
    env.close()
