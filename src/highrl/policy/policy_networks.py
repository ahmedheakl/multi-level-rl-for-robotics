"""Policy networks for the teacher implementation using the actor crtic module"""
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym import spaces
from torch import nn
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class TeacherPolicyNetwork(nn.Module):
    """Teacher network for policy and value function.
    It receives as input the features extracted by the feature extractor, and outputs
    the difficulty value of the policy network, and a vector representing the value for
    the value network.

    Args:
        feature_dim (int, optional): Dimension of the features extracted with the
        features_extractor (e.g. features from a CNN). Defaults to 4.
        last_layer_dim_pi: (int, optional): Number of units for the last layer of
        the policy network. Defaults to 1.
        last_layer_dim_vf: (int, optional): Number of units for the last layer of
        the value network. Detaults to 8.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 8,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 8),
            nn.ReLU(),
            nn.Linear(feature_dim * 8, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, last_layer_dim_pi),
            nn.Sigmoid(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 16),
            nn.ReLU(),
            nn.Linear(feature_dim * 16, feature_dim * 8),
            nn.ReLU(),
            nn.Linear(feature_dim * 8, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass through both the policy network and the value network.

        Args:
            features (th.Tensor): Features from the feature extractor module

        Returns:
            Tuple[th.Tensor, th.Tensor]: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``.
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through the actor network"""
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forawrd pass through the value network"""
        return self.value_net(features)


class TeacherActorCriticPolicy(ActorCriticPolicy):
    """Model implementation for actor/critic policy"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TeacherPolicyNetwork()
