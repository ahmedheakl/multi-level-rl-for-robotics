import gym
import torch.nn as nn
import torch as th
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy


class LinearPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 16,
        last_layer_dim_pi: int = 7,
        last_layer_dim_vf: int = 32,
    ):
        super(LinearPolicyNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class LSTMPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 16,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 32,
        max_big_obs: int = 2,
        max_med_obs: int = 5,
        max_small_obs: int = 7,
        hidden_dim: int = 16,
    ):
        super(LSTMPolicyNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.max_big_obs = max_big_obs
        self.max_med_obs = max_med_obs
        self.max_small_obs = max_small_obs
        self.max_num_obs = max_big_obs + max_med_obs + max_small_obs

        self.embedding = nn.Linear(last_layer_dim_pi, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, last_layer_dim_pi)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.hidden = th.zeros(1, 1, hidden_dim, device=self.device)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, last_layer_dim_vf),
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
        self, input_tensor: th.Tensor, hidden_tensor: th.Tensor
    ) -> Tuple[th.Tensor, ...]:
        """Foward pass through decoder policy net

        Args:
            input_tensor (th.Tensor): <SOS> for _first layer_, and output from last layer for _other layers_
            hidden_tensor (th.Tensor): features for _first layer_, and hidden from last layer for _other layers_

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
        decoder_input = th.zeros(1, 4).to(self.device)  # <SOS>

        # first output = [px, py, gx, gy]
        # input_dims : [1, 4], [1, 16]
        # output_dims: [1, 4], [1, 16], [1, 4]
        p_out, hidden, logits = self.decoder_policy(decoder_input, features)
        robot_pos = self.softmax(logits[0])  # shape: [4]

        # second output = [big_sz, med_sz, sm_sz, _]
        p_out, hidden, logits = self.decoder_policy(p_out, hidden)
        obs_count = self.softmax(logits[0])

        num_big_obs = th.round(obs_count[0] * self.max_big_obs).int().item()
        num_med_obs = th.round(obs_count[1] * self.max_med_obs).int().item()
        num_sm_obs = th.round(obs_count[2] * self.max_small_obs).int().item()

        target_length = num_big_obs + num_med_obs + num_sm_obs
        outputs = th.zeros(self.max_num_obs + 2, 1, self.latent_dim_pi).to(self.device)

        outputs[0] = robot_pos
        outputs[1] = obs_count

        for t in range(2, target_length + 2):
            p_out, hidden, logits = self.decoder_policy(p_out, hidden)
            outputs[t] = self.softmax(logits[0])
        print("outputs shape = ", outputs.shape)
        return outputs

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class LinearActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(LinearActorCriticPolicy, self).__init__(
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

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMPolicyNetwork()
