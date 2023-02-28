"""Implementation for the evaluation environemnt"""
from typing import List, Tuple
from random import randint
from argparse import Namespace
from configparser import RawConfigParser

from highrl.utils import Position
from highrl.envs.robot_env import RobotEnv
from highrl.obstacle.single_obstacle import SingleObstacle


class RobotEvalEnv(RobotEnv):
    """Evaluation environment for robot performance"""

    def __init__(self, config: RawConfigParser, args: Namespace) -> None:
        super().__init__(config, args)
        self.cfg.n_eval_episodes = config.getint("eval", "n_eval_episodes")

        self.set_robot_position(
            Position[int](self.cfg.robot_init_x_pos, self.cfg.robot_init_y_pos),
            Position[int](self.cfg.goal_x_pos, self.cfg.goal_y_pos),
        )

        big_obs_pos = [(180, 180), (125, 125)]
        med_obs_pos = [(125, 70), (70, 125), (200, 50), (150, 30)]
        sml_obs_pos = [(30, 10), (10, 50), (100, 80), (20, 170)]

        obstacles = self.generate_eval_obstacles(
            self.cfg.eval_big_obs_count,
            self.cfg.eval_med_obs_count,
            self.cfg.eval_sml_obs_count,
            big_obs_pos,
            med_obs_pos,
            sml_obs_pos,
            self.cfg.eval_big_obs_dim,
            self.cfg.eval_med_obs_dim,
            self.cfg.eval_sml_obs_dim,
        )

        for obstacle in obstacles:
            self.obstacles.obstacles_list.append(obstacle)

    def generate_eval_obstacles(
        self,
        big_obs_count: int,
        med_obs_count: int,
        sml_obs_count: int,
        big_obs_pos: List[Tuple[int, int]],
        med_obs_pos: List[Tuple[int, int]],
        sml_obs_pos: List[Tuple[int, int]],
        big_obs_dim: int,
        med_obs_dim: int,
        sml_obs_dim: int,
    ) -> List[SingleObstacle]:
        """Generates obstacles for the evaluation environment

        Args:
            big_obs_count (int): number of big type obstacles
            med_obs_count (int): number of medium type obstacles
            sml_obs_count (int): number of small type obstacles
            big_obs_pos (List[Tuple[int, int]]): List cotaining a tuple of x and y positions for every big typed obstacles
            med_obs_pos (List[Tuple[int, int]]): List cotaining a tuple of x and y positions for every medium typed obstacles
            sml_obs_pos (List[Tuple[int, int]]): List cotaining a tuple of x and y positions for every small typed obstacles
            big_obs_dim (int): size of big obstacles
            med_obs_dim (int): size of medium obstacles
            sml_obs_dim (int): size of small obstacles

        Returns:
            List[SingleObstacle]: List of obstacles for the evaluation environment
        """
        obstacles = []
        for big_obs_index in range(big_obs_count):
            obs_x = big_obs_pos[big_obs_index][0]
            obs_y = big_obs_pos[big_obs_index][1]
            obs_w = big_obs_dim
            obs_h = big_obs_dim
            obstacles.append(SingleObstacle(obs_x, obs_y, obs_w, obs_h))

        for med_obs_index in range(med_obs_count):
            obs_x = med_obs_pos[med_obs_index][0]
            obs_y = med_obs_pos[med_obs_index][1]
            obs_w = med_obs_dim
            obs_h = med_obs_dim
            obstacles.append(SingleObstacle(obs_x, obs_y, obs_w, obs_h))

        for sml_obs_index in range(sml_obs_count):
            obs_x = sml_obs_pos[sml_obs_index][0]
            obs_y = sml_obs_pos[sml_obs_index][1]
            obs_w = sml_obs_dim
            obs_h = sml_obs_dim
            obstacles.append(SingleObstacle(obs_x, obs_y, obs_w, obs_h))
        return obstacles
