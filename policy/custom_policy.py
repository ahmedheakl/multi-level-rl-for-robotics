from turtle import done
import torch as th
import gym
import torch.nn as nn


from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


GREEN_COLOR = (50, 225, 30)
BLUE_COLOR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MIN_SCREEN_WIDTH = 200
MIN_SCREEN_HEIGHT = 200


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 37):
        super(CustomFeatureExtractor, self).__init__(
            observation_space=observation_space, features_dim=features_dim
        )

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
        lidar_obs = observations["lidar"]
        rs_obs = observations["robot"]
        lidar_obs = th.unsqueeze(lidar_obs, dim=1)
        return th.cat((self.cnn(lidar_obs), rs_obs), axis=1)


# if __name__ == "__main__":
#     policy_kwargs = dict(
#         features_extractor_class=CustomFeatureExtractor,
#         net_arch=[dict(vf=[256], pi=[16])],

#     )
#     env = TestEnv()
#     model = PPO("CnnPolicy", "TestEnv", policy_kwargs=policy_kwargs, verbose=1)
#     model.learn(1000)


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
        for key, subspace in observation_space.spaces.items():
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
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
