"""Gym registration for the guarded P5 pursuit task."""

import gymnasium as gym


gym.register(
    id="Bylw-Pursuit-Direct-v0",
    entry_point=f"{__name__}.bylw_pursuit_env:BylwPursuitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.bylw_pursuit_env_cfg:BylwPursuitEnvCfg"
        ),
    },
)

gym.register(
    id="Bylw-Pursuit-Native-Train-v0",
    entry_point=f"{__name__}.bylw_pursuit_train_env:BylwPursuitTrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.bylw_pursuit_train_env_cfg:BylwPursuitTrainEnvCfg"
        ),
    },
)
