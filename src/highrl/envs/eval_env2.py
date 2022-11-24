from typing import List, Tuple
from highrl.envs.robot_env import RobotEnv
from highrl.obstacle.single_obstacle import SingleObstacle
import numpy as np
from random import randint, random, uniform
from time import time
import configparser
from highrl.envs.env_encoders import RobotEnv1DPlayer, RobotEnv2DPlayer
import pandas as pd
import math
from os import path
import argparse
from gym import Env, spaces
from highrl.obstacle.obstacles import Obstacles
from highrl.agents.robot import Robot
from typing import Any, List
from gym import Env, spaces
from highrl.utils.action import *
from highrl.obstacle.obstacles import Obstacles
import numpy as np
from CMap2D import render_contours_in_lidar

from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from highrl.utils.calculations import *
import threading
from highrl.agents.robot import Robot
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.configs.colors import *
import configparser
import pandas as pd
import time
import argparse
from os import path

class RobotEvalEnv(Env):
    def __init__(
        self, config: configparser.RawConfigParser, args: argparse.Namespace
    ) -> None:
        super(RobotEvalEnv, self).__init__()
        self.action_space_names = ["ActionXY", "ActionRot"]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32
                ),
                "robot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                ),
            }
        )

        self.obstacles = Obstacles()
        self.robot = Robot()
        self.viewer = None
        self.args = args
        self.reward = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_reward = 0
        self.total_steps = 0
        self.success_flag = False

        self.done = False

        self.robot_initial_px = 0
        self.robot_initial_py = 0
        self.robot_goal_px = 0
        self.robot_goal_py = 0

        self.is_initial_state = True

        """Results of each episode
        Contains [episode_reward, episode_steps, success_flag]
        """
        self.results = []

        self._configure(config=config)

        self.episode_statistics = None
        if self.collect_statistics:
            self.episode_statistics = pd.DataFrame(
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

    
    def _configure(self, config: configparser.RawConfigParser) -> None:
        """Configure the environment using input config file

        Args:
            config (configparser.RawConfigParser): input config object
        """
        self.config = config
        self.robot_initial_px = config.getint("positions", "robot_initial_px")
        self.robot_initial_py =  config.getint("positions", "robot_initial_py")
        self.robot_goal_px = config.getint("positions", "robot_goal_px")
        self.robot_goal_py = config.getint("positions", "robot_goal_py")
        self.width = config.getint("dimensions", "width")
        self.height = config.getint("dimensions", "height")
        self.robot_radius = config.getint("dimensions", "robot_radius")
        self.goal_radius = config.getint("dimensions", "goal_radius")
        
        self.num_of_hard_obstacles = config.getint("obstacles", "n_hard")
        self.num_of_medium_obstacles = config.getint("obstacles", "n_,medium")
        self.num_of_small_obstacles = config.getint("obstacles", "n_small")
        self.delta_t = config.getint("timesteps", "delta_t")
        self.max_episode_steps = config.getint("timesteps", "max_episode_steps")

        self.n_angles = config.getint("lidar", "n_angles")
        self.lidar_angle_increment = config.getfloat("lidar", "lidar_angle_increment")
        self.lidar_min_angle = config.getfloat("lidar", "lidar_min_angle")
        self.lidar_max_angle = config.getfloat("lidar", "lidar_max_angle")

        self.collision_score = config.getint("reward", "collision_score")
        self.reached_goal_score = config.getint("reward", "reached_goal_score")
        self.minimum_velocity = config.getfloat("reward", "minimum_velocity")
        self.minimum_distance = config.getfloat("reward", "minimum_distance")
        self.maximum_distance = config.getfloat("reward", "maximum_distance")
        self.velocity_std = config.getfloat("reward", "velocity_std")
        self.alpha = config.getfloat("reward", "alpha")

        self.render_each = config.getint("render", "render_each")
        self.save_to_file = config.getboolean("render", "save_to_file")

        self.epsilon = config.getint("env", "epsilon")
        self.collect_statistics = config.getboolean("statistics", "collect_statistics")
        self.scenario = config.get("statistics", "scenario")
    
            
    def obstacles_configure (self) -> None:
        self._generate_obstacles_points(
           self.num_of_hard_obstacles,
            min_dim=300,
            max_dim=500,
        )
        self._generate_obstacles_points(
           self.num_of_medium_obstacles ,
            min_dim=100,
            max_dim=250,
        )
        self._generate_obstacles_points(
            self.num_of_small_obstacles,
            min_dim=10,
            max_dim=50,
        )

    def _generate_obstacles_points(
        self, obstacles_count: int, min_dim: int, max_dim: int
    ) -> None:
        """Generate obstacles based on teacher action for next robot session

        Args:
            obstacles_count (int): number of obstacles
        """
        self.add_boarder_obstacles()
        for i in range(int(obstacles_count)):
            overlap = True
            new_obstacle = SingleObstacle() 
            while overlap:
                px = randint(0, self.width)
                py = randint(0, self.height)
                new_width = randint(min_dim, max_dim)
                new_height = randint(min_dim, max_dim)
                new_obstacle = SingleObstacle(px, py, new_width, new_height)
                overlap = self.robot.is_overlapped(
                    new_obstacle, check_target="robot"
                ) or self.robot.is_overlapped(
                    new_obstacle, check_target="goal"
                )
            self.obstacles += new_obstacle


            



if __name__ == "__main__":
    from highrl.utils.parser import parse_args, generate_agents_config, handle_output_dir
    args = parse_args()
    #print(args)
    robot_config, teacher_config = generate_agents_config(
            args.robot_config_path, args.teacher_config_path
        )
    #print(robot_config)
    env = RobotEvalEnv(robot_config, args)
    player = RobotEnv2DPlayer(env,args)

































