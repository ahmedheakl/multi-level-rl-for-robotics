"""Implementation for the policy networks"""
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import logging
from dataclasses import dataclass
from gym import spaces
from torch import nn
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from highrl.utils.utils import get_device

_LOG = logging.getLogger(__name__)


class LinearPolicyNetwork(nn.Module):
    """Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        last_layer_dim_pi: int = 8,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        inner_dim = 32
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, inner_dim * 4),
            nn.ReLU(),
            nn.Linear(inner_dim * 4, inner_dim * 2),
            nn.ReLU(),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, last_layer_dim_pi),
            nn.Sigmoid(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, inner_dim * 4),
            nn.ReLU(),
            nn.Linear(inner_dim * 4, inner_dim * 2),
            nn.ReLU(),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass through both actor/critic networks"""
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through the actor network"""
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through the critic network"""
        return self.value_net(features)


@dataclass
class GRUDims:
    """Dimension variables for GRU model"""

    max_big_obs: int = 2
    max_med_obs: int = 5
    max_small_obs: int = 7

    def get_max_num_obs(self) -> int:
        """Retrieve the sum of obstacles"""
        return self.max_big_obs + self.max_med_obs + self.max_small_obs


class GRUPolicyNetwork(nn.Module):
    """Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    """

    hidden_dim = 16
    latent_dim_pi = 4
    latent_dim_vf = 32

    def __init__(
        self,
        feature_dim: int = 16,
        max_big_obs: int = 2,
        max_med_obs: int = 5,
        max_small_obs: int = 7,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.net_dims = GRUDims(max_big_obs, max_med_obs, max_small_obs)
        self.embedding = nn.Linear(self.latent_dim_pi, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.latent_dim_pi)
        self.softmax = nn.Softmax(dim=0)
        self.hidden = th.zeros(1, 1, self.hidden_dim, device=get_device())
        inner_dim = 32
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, inner_dim * 4),
            nn.ReLU(),
            nn.Linear(inner_dim * 4, inner_dim * 2),
            nn.ReLU(),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass for both actor and critic

        Args:
            features (th.Tensor): features vector from feature extractor net

        Returns:
            Tuple[th.Tensor, th.Tensor]: action(4), value(1, 1, 32)
        """
        return self.forward_actor(features), self.value_net(features)

    def decoder_policy(
        self,
        input_tensor: th.Tensor,
        hidden_tensor: th.Tensor,
    ) -> Tuple[th.Tensor, ...]:
        """Foward pass through decoder policy net

        Args:
            input_tensor (th.Tensor): <SOS> for _first layer_, and output
            from last layer for _other layers_
            hidden_tensor (th.Tensor): features for _first layer_, and hidden
            from last layer for _other layers_

        Returns:
           Tuple[th.Tensor, ...] : output(1, 4), hidden(1, 16), logits(1, 4): un-normalized outputs
        """
        # justify dims [1, 4], [1, 16]
        input_tensor = input_tensor.view(1, -1)
        hidden_tensor = hidden_tensor.view(1, -1)

        embedded = self.embedding(input_tensor)
        embedded = nn.functional.relu(embedded)
        output, hidden_tensor = self.gru(embedded, hidden_tensor)
        logits = self.out(output)

        # justify output: [1, 4]
        output = logits.view(1, -1)
        return output, hidden_tensor, logits

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Full pass through the actor net

        Args:
            features (th.Tensor): features from feature extractor net

        Returns:
            th.Tensor: outputs(num_obs + 2, 4)
        """
        decoder_input = th.zeros(1, 4).to(get_device())  # <SOS>

        # first output = [px, py, gx, gy]
        # input_dims : [1, 4], [1, 16]
        # output_dims: [1, 4], [1, 16], [1, 4]
        p_out, hidden, logits = self.decoder_policy(decoder_input, features)
        robot_pos = self.softmax(logits[0])  # shape: [4]

        # second output = [big_sz, med_sz, sm_sz, _]
        p_out, hidden, logits = self.decoder_policy(p_out, hidden)
        obs_count = self.softmax(logits[0])

        num_big_obs = int(th.round(obs_count[0] * self.net_dims.max_big_obs).item())
        num_med_obs = int(th.round(obs_count[1] * self.net_dims.max_med_obs).item())
        num_sm_obs = int(th.round(obs_count[2] * self.net_dims.max_small_obs).item())

        target_length = num_big_obs + num_med_obs + num_sm_obs
        outputs = th.zeros(
            self.net_dims.get_max_num_obs() + 2,
            1,
            self.latent_dim_pi,
        )
        outputs = outputs.to(get_device())

        outputs[0] = robot_pos
        outputs[1] = obs_count

        for target_index in range(2, target_length + 2):
            p_out, hidden, logits = self.decoder_policy(p_out, hidden)
            outputs[target_index] = self.softmax(logits[0])
        output_str = f"Action shape: {outputs.shape}"
        _LOG.debug(output_str)
        return outputs

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through the critic network"""
        return self.value_net(features)


class GRUActorCriticPolicy(ActorCriticPolicy):
    """ActorCritic policy implementation using a linear model"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
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
        # disable orthogonal initialization
        self.ortho_init = False
        self._build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GRUPolicyNetwork()
