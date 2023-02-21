"""Utilties for robot training"""
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from highrl.utils.utils import Position


@dataclass
class RobotOpt:
    """Optimizers for robot environment"""

    reward: float = 0
    episodes = 1
    episode_reward: float = 0
    episode_steps: int = 0
    total_reward: float = 0
    total_steps: int = 0
    success_flag: bool = False
    num_successes: int = 0

    robot_init_pos: Position = Position()
    goal_init_pos: Position = Position()

    is_initial_state: bool = True

    # Results of each episode
    # Contains [episode_reward, episode_steps, success_flag]
    results: List[Tuple[float, int, bool]] = [(0.0, 0, False)]

    # lidar_scan: np.ndarray = np.array([])
    # lidar_angles: np.ndarray = np.array([])

    # contours: np.ndarray = np.array([])
    # flat_contours: np.ndarray = np.array([])

    def set_tb_writer(self, path_to_events) -> None:
        """Setter for robot writer"""
        self.tb_writer = SummaryWriter(path_to_events)


if __name__ == "__main__":
    opt = RobotOpt()

    episode_statistics = pd.DataFrame(
        columns=[
            "total_steps",
            "episode_steps",
            "scenario",
            "damage",
            "goal_reached",
            "total_reward",
            "episode_reward",
            "reward",
            "wall_time",
        ]
    )
    tb_writer: SummaryWriter = SummaryWriter("runs")
