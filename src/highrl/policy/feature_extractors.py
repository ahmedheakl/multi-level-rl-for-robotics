"""
Implementation of features exctractors for both the teacher and the robot
"""
import logging
from gym import spaces
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_LOG = logging.getLogger(__name__)



class TeacherFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for the teacher implemented using LSTM.

    The model takes in for each step:
        (1) Number of sucesses for the last robot session
        (2) Robot average reward for the last robot session
        (3) Average number of steps per episode for the last robot session
        (4) Robot level of the upcoming session

    and produces the difficulty for the upcoming session.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 4,
        hidden_size: int = 16,
        num_layers: int = 1,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Dim = [num_layers, batch_size, hidden_size]
        self.hidden = th.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size,
            device=self.device,
        )

        # Dim = [num_layers, batch_size, hidden_size]
        self.cell = th.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size,
            device=self.device,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass through the feature extractor

        Observations dim = [batch_size, features_dim]
        """
        # Re-initialize hidden state if the robot level is reset
        for batch_idx, observ in enumerate(observations):
            if observ[-1] >= 1.0:
                continue

            self.hidden[:, batch_idx, :] = th.zeros(
                self.num_layers,
                self.hidden_size,
                device=self.device,
            )
            self.cell[:, batch_idx, :] = th.zeros(
                self.num_layers,
                self.hidden_size,
                device=self.device,
            )

        # Adding sequence length dimension
        # Dim = [seq_len, batch_size, features_dim]
        observations = observations.unsqueeze(0)

        _LOG.debug("Observation size: %s", observations.shape)
        _LOG.debug("Hidden size: %s", self.hidden.shape)
        _LOG.debug("Cell size: %s", self.cell.shape)

        # Dim = [seq_len, batch_size, input_size]
        # Here is seq_len=1, since we generating a new environment
        # for every single observation.
        observations = observations.to(self.device)
        output_tensor, (self.hidden, self.cell) = self.lstm(
            observations,
            (self.hidden, self.cell),
        )
        _LOG.debug("Output teacher features size: %s", output_tensor.shape)
        assert output_tensor.shape == th.Size([1, self.batch_size, self.hidden_size])

        output_tensor: th.Tensor = output_tensor.squeeze(0)

        # Return shape = [batch_size, hidden_size]
        return output_tensor


class Robot2DFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 37):
        super().__init__(observation_space=observation_space, features_dim=features_dim)

        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 32),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lidar_obs = observations["lidar"]  # type: ignore
        rs_obs = observations["robot"]  # type: ignore
        lidar_obs = th.unsqueeze(lidar_obs, dim=1)
        return th.cat((self.cnn(lidar_obs), rs_obs), axis=1)  # type: ignore


class Robot1DFeatureExtractor(BaseFeaturesExtractor):
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

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lidar_obs = observations["lidar"]  # type: ignore
        rs_obs = observations["robot"]  # type: ignore
        lidar_obs = th.unsqueeze(lidar_obs, dim=1)
        return th.cat((self.cnn(lidar_obs), rs_obs), axis=1)  # type: ignore
