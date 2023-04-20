"""Implementation of environment generator using RL"""
from typing import Tuple, Callable, List
import logging
import random

import numpy as np
from gym import Env, spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from stable_baselines3 import PPO  # type: ignore
import torch
from torch import nn

from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.agents.robot import Robot
from highrl.utils import Position
from highrl.utils.teacher_checker import compute_difficulty
from highrl.envs.env_generator import EnvGeneratorModelLSTM


_LOG = logging.getLogger(__name__)


HARD_OBS = 2
MED_OBS = 3
SMALL_OBS = 4
OBS_CNT = HARD_OBS + MED_OBS + SMALL_OBS
HARD_SIZE = 50
MED_SIZE = 40
SMALL_SIZE = 30
OUTPUT_SIZE = 4
ENV_SIZE = 256
MAX_DIFFICULTY = ENV_SIZE * ENV_SIZE
NUM_POINTS: int = OUTPUT_SIZE * (OBS_CNT + 1)
DEVICE = "cuda"


class EnvGeneratorPolicyLSTM(nn.Module):
    """Implementation for the environment generator model"""

    num_layers = 1

    def __init__(
        self,
        feature_dim: int = 1,
        last_layer_dim_pi: int = 40,
        last_layer_dim_vf: int = 32,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.hidden_size = hidden_size

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.ReLU(),
        )

        self.policy_net = EnvGeneratorModelLSTM(self.hidden_size, self.num_layers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the value network and policy network"""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network"""
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network"""
        return self.value_net(features)


class EnvGeneratorActorCritic(ActorCriticPolicy):
    """Actor critic architecture implementation for the environment
    generation model"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self.mlp_extractor = EnvGeneratorPolicyLSTM()


class GeneratorEnv(Env):
    """Gym environment implementation to train the environment generator model"""

    def __init__(self, max_obstacles: int = 10):
        super().__init__()
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 * max_obstacles,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=MAX_DIFFICULTY,
            shape=(1,),
            dtype=np.float32,
        )
        self.difficulty: float = 0
        self.time_step: int = 0
        self.tb_writer = SummaryWriter("runs")

    def step(self, action: np.ndarray) -> Tuple[List[float], float, bool, dict]:
        """Step through the environment and return the next observation

        Action shape = (40, ). All values are between [0, 1]
        """
        robot_pos = action[:4]
        rob_x = robot_pos[0] * ENV_SIZE
        rob_y = robot_pos[1] * ENV_SIZE
        goal_x = robot_pos[2] * ENV_SIZE
        goal_y = robot_pos[3] * ENV_SIZE
        robot = Robot(Position[float](rob_x, rob_y), Position[float](goal_x, goal_y))
        obstacles_ls = []

        # Convert obstacles position/dimension from [0, 1] to [0, width]
        for idx in range(4, 40, 4):
            dims = [action[idx + dim_i] * ENV_SIZE for dim_i in range(4)]
            obstacles_ls.append(SingleObstacle(*dims))
        obstacles = Obstacles(obstacles_ls)

        # Compute the difficulty of the generated environment
        old_difficulty = self.difficulty
        self.difficulty, _ = compute_difficulty(
            obstacles,
            robot,
            ENV_SIZE,
            ENV_SIZE,
        )
        reward: float = -abs(self.difficulty - old_difficulty)
        self.tb_writer.add_scalar("generator_reward", reward, self.time_step)
        self.time_step += 1
        _LOG.info("Reward %f", reward)

        # Notice here we are returning done=True, so that the model would update
        # its weights after each step, as it is considering each step as separate
        # episode
        return self._make_obs(), reward, True, {}

    def _make_obs(self) -> List[float]:
        """Create observations for the environment"""

        # Generating a new random difficulty value as an observation for the model
        self.difficulty = random.random() * MAX_DIFFICULTY
        return [self.difficulty]

    def reset(self):
        """Reset environemnt and return start observation"""
        return self._make_obs()

    # pylint: disable=arguments-differ
    def render(self):
        """Overriding render method"""
        return super().render()


def train_rl() -> None:
    """Main method for starting the training for the envirioment
    generator model"""
    env = GeneratorEnv(OBS_CNT + 1)
    model = PPO(EnvGeneratorActorCritic, env, verbose=1, device=DEVICE)
    model.learn(5000)
    model.save("generator_model/")
