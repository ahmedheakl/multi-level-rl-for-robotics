"""Implementation of feature extractors for both robot and teacher"""
from typing import Dict
from gym import spaces
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from highrl.utils.utils import get_device


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor implementation for the teacher using GRU sequential model"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 6,
        hidden_dim: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.hidden_dim = hidden_dim
        self.init_hidden()
        self.linear = nn.Linear(features_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)

    def init_hidden(self) -> None:
        """Initialize hidden vector"""
        self.hidden = th.zeros(1, 1, self.hidden_dim, device=get_device())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass through the feature extractor.

        The forward starts with providing the current ovbservation as an input
        to the GRU, taking the hidden state as the output of the current state.

        Args:
            observations (th.Tensor): Observations as input for the current state

        Returns:
            th.Tensor: Hidden state of the GRU model
        """

        observations.clone().detach()

        # if robot level is reset, then re-initialize hidden tensor
        if observations[0][0] == 0:
            self.init_hidden()

        embedded = self.linear(observations).view(1, 1, -1)
        _, hidden = self.gru(embedded, self.hidden)

        # returns [1, 1, 16]
        return hidden


class Robot2DFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the robot using rings lidar readings"""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 37):
        super().__init__(observation_space=observation_space, features_dim=features_dim)
        feature_dim = 32
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(features_dim, feature_dim * 2, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(feature_dim * 8, feature_dim * 8, kernel_size=1, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 32),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Forward pass through the feature extractor

        Args:
            observations (th.Tensor): Observations tensor

        Returns:
            th.Tensor: Processed lidar observations concatenated with the robot observations
        """
        lidar_obs: th.Tensor = observations["lidar"]
        rs_obs: th.Tensor = observations["robot"]
        lidar_obs = th.unsqueeze(lidar_obs, dim=1)
        processed_state: th.Tensor = self.cnn(lidar_obs)
        return th.cat((processed_state, rs_obs), 1)


class Robot1DFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the robot using flat lidar readings"""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 37):
        super().__init__(observation_space=observation_space, features_dim=features_dim)

        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 32),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Forward pass through the feature extractor

        Args:
            observations (th.Tensor): Observations tensor

        Returns:
            th.Tensor: Processed lidar observations concatenated with the robot observations
        """
        lidar_obs: th.Tensor = observations["lidar"]
        rs_obs: th.Tensor = observations["robot"]
        lidar_obs = th.unsqueeze(lidar_obs, dim=1)
        processed_state = self.cnn(lidar_obs)
        return th.cat((processed_state, rs_obs), 1)
