"""Implementation of feature extractors for both robot and teacher"""
import gym
from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from highrl.utils.utils import get_device


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
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

        observations.clone().detach()

        # if robot level is reset, then re-initialize hidden tensor
        # FIXME: check robot level entry position
        print("observations = ", observations)
        if observations[0][0] == 0:
            self.init_hidden()

        embedded = self.linear(observations).view(1, 1, -1)
        _, self.hidden = self.gru(embedded, self.hidden)

        # returns [1, 1, 16]
        print("features shape = ", self.hidden.shape)
        return self.hidden


class Robot2DFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 37):
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
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 37):
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


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():  # type: ignore
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))  # type: ignore
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
