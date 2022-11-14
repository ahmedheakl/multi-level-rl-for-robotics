from gym import spaces
import numpy as np
from highrl.lidar_setup.rings import generate_rings
from highrl.envs.robot_env import RobotEnv
import configparser

_L = 1080  # lidar size
_RS = 5  # robotstate size


class FlatLidarEncoder:
    """Genetric class to encode environment for 1D lidar states"""

    def __init__(self) -> None:
        self.lidar_dim = 1080
        self.robot_dim = 5
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.lidar_dim,), dtype=np.float32
                ),
                "robot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.robot_dim,), dtype=np.float32
                ),
            }
        )

    def _encode_obs(self, obs: dict) -> dict:
        return obs


class RingsLidarEncoder:
    """Genetric class to encode environment for 2D lidar states
    Assumes usage of 1D Conv
    """

    def __init__(self) -> None:
        self.ring_dim = 64 * 64
        self.robot_dim = 5
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.ring_dim,), dtype=np.float32
                ),
                "robot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.robot_dim,), dtype=np.float32
                ),
            }
        )

        self.rings_def = generate_rings(64, 64)

    def _encode_obs(self, obs: dict) -> dict:
        lidar = obs["lidar"]
        obs["lidar"] = (
            self.rings_def["lidar_to_rings"](lidar[None, :]).astype(float)
            / self.rings_def["rings_to_bool"]
        ).reshape(self.ring_dim)
        return obs


class RobotEnv1DPlayer(RobotEnv):
    def __init__(self, config: configparser.RawConfigParser) -> None:
        self.encoder = FlatLidarEncoder()
        super().__init__(config)

        self.observation_space = self.encoder.observation_space

    def step(self, action):
        obs, reward, done, info = super(RobotEnv1DPlayer, self).step(action)
        h = self.encoder._encode_obs(obs)

        return h, reward, done, info

    def reset(self):
        obs = super(RobotEnv1DPlayer, self).reset()
        h = self.encoder._encode_obs(obs)
        return h


class RobotEnv2DPlayer(RobotEnv):
    def __init__(self, config: configparser.RawConfigParser) -> None:
        self.encoder = FlatLidarEncoder()
        super().__init__(config)

        self.observation_space = self.encoder.observation_space

    def step(self, action):
        obs, reward, done, info = super(RobotEnv2DPlayer, self).step(action)
        h = self.encoder._encode_obs(obs)

        return h, reward, done, info

    def reset(self):
        obs = super(RobotEnv2DPlayer, self).reset()
        h = self.encoder._encode_obs(obs)
        return h
