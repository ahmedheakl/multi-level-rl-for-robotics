"""Implementation for environments wrappers"""
import configparser
import argparse
from typing import List, Tuple
from gym import spaces
import numpy as np

from highrl.lidar_setup.rings import generate_rings
from highrl.envs.robot_env import RobotEnv
from highrl.envs.eval_env import RobotEvalEnv


class FlatLidarEncoder:
    """Generic class to encode environment for 1D lidar states"""

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

    def encode_obs(self, obs: dict) -> dict:
        """Encode observations from robot lidar

        Note that the function is idle since the lidar readings
        are inherently in the 1-D format.

        Args:
            obs (dict): Input observation for encoding

        Returns:
            dict: Encoded observation
        """
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

    def encode_obs(self, obs: dict) -> dict:
        """Encode observations from robot lidar

        Convert observations to 2-D format.

        Args:
            obs (dict): Input observation for encoding

        Returns:
            dict: Encoded observation
        """
        lidar = obs["lidar"]
        obs["lidar"] = (
            self.rings_def["lidar_to_rings"](lidar[None, :]).astype(float)
            / self.rings_def["rings_to_bool"]
        ).reshape(self.ring_dim)
        return obs


class RobotEnv1DPlayer(RobotEnv):
    """Robot environment wrapper for running flat lidars readings"""

    def __init__(
        self,
        config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        self.encoder = FlatLidarEncoder()
        super().__init__(config, args)
        self.observation_space = self.encoder.observation_space

    def step(self, action: List) -> Tuple[dict, float, bool, dict]:
        """Implements one step for robot

        Args:
            action (List): Action for robot step

        Returns:
            Tuple[dict, float, bool, dict]: observation, reward, done flag, info dict
        """
        obs, reward, done, info = super().step(action)
        encoded_obs = self.encoder.encode_obs(obs)

        return encoded_obs, reward, done, info

    def reset(self) -> dict:
        """Reset robot environment

        Returns:
            dict: Observation
        """
        obs = super().reset()
        encoded_obs = self.encoder.encode_obs(obs)
        return encoded_obs


class RobotEnv2DPlayer(RobotEnv):
    """Robot environment wrapper for running rings lidars readings"""

    def __init__(
        self,
        config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        self.encoder = RingsLidarEncoder()  # need to be changed
        super().__init__(config, args)
        self.observation_space = self.encoder.observation_space

    def step(self, action: List) -> Tuple[dict, float, bool, dict]:
        """Implements one step for robot

        Args:
            action (List): Action for robot step

        Returns:
            Tuple[dict, float, bool, dict]: observation, reward, done flag, info dict
        """
        obs, reward, done, info = super().step(action)
        encoded_obs = self.encoder.encode_obs(obs)

        return encoded_obs, reward, done, info

    def reset(self) -> dict:
        """Reset robot environment

        Returns:
            dict: Observation
        """
        obs = super().reset()
        encoded_obs = self.encoder.encode_obs(obs)
        return encoded_obs


class EvalEnv1DPlayer(RobotEvalEnv):
    """Robot evaluatiion environment wrapper for running flat lidars readings"""

    def __init__(
        self,
        config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        self.encoder = FlatLidarEncoder()
        super().__init__(config, args)

        self.observation_space = self.encoder.observation_space

    def step(self, action: List) -> Tuple[dict, float, bool, dict]:
        """Implements one step for robot

        Args:
            action (List): Action for robot step

        Returns:
            Tuple[dict, float, bool, dict]: observation, reward, done flag, info dict
        """
        obs, reward, done, info = super().step(action)
        encoded_obs = self.encoder.encode_obs(obs)

        return encoded_obs, reward, done, info

    def reset(self) -> dict:
        """Reset robot environment

        Returns:
            dict: Observation
        """
        obs = super().reset()
        encoded_obs = self.encoder.encode_obs(obs)
        return encoded_obs


class EvalEnv2DPlayer(RobotEvalEnv):
    """Robot evaluatiion environment wrapper for running rings lidars readings"""

    def __init__(
        self,
        config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        self.encoder = RingsLidarEncoder()  # need to be changed
        super().__init__(config, args)

        self.observation_space = self.encoder.observation_space

    def step(self, action: List) -> Tuple[dict, float, bool, dict]:
        """Implements one step for robot

        Args:
            action (List): Action for robot step

        Returns:
            Tuple[dict, float, bool, dict]: observation, reward, done flag, info dict
        """
        obs, reward, done, info = super().step(action)
        encoded_obs = self.encoder.encode_obs(obs)

        return encoded_obs, reward, done, info

    def reset(self) -> dict:
        """Reset robot environment

        Returns:
            dict: Observation
        """
        obs = super().reset()
        encoded_obs = self.encoder.encode_obs(obs)
        return encoded_obs
