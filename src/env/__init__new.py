from gymnasium.envs.registration import register

register(
    id='pacman-v0',
    entry_point='src.env.pacman_env_new:PacmanEnv',
    max_episode_steps=10000,
    kwargs={'layout': 'classic'}
)
