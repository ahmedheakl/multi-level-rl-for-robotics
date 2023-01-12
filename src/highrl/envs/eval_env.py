from typing import List, Tuple
from highrl.envs.robot_env import RobotEnv
from highrl.obstacle.single_obstacle import SingleObstacle
from random import randint, random, uniform
import configparser
from os import path


class RobotEvalEnv(RobotEnv):
    def __init__(self, *args, **kwargs):
        super(RobotEvalEnv, self).__init__(*args, **kwargs)
        self.obstacles_configure()

    def _configure(self, config: configparser.RawConfigParser) -> None:
        """Configure the environment using input config file

        Args:
            config (configparser.RawConfigParser): input config object
        """
        self.config = config
        self.robot_initial_px = config.getint("positions", "robot_initial_px")
        self.robot_initial_py = config.getint("positions", "robot_initial_py")
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
        self.progress_discount = config.getfloat("reward", "progress_discount")

        self.render_each = config.getint("render", "render_each")
        self.save_to_file = config.getboolean("render", "save_to_file")

        self.epsilon = config.getint("env", "epsilon")
        self.collect_statistics = config.getboolean("statistics", "collect_statistics")
        self.scenario = config.get("statistics", "scenario")

    def obstacles_configure(self) -> None:
        self._generate_obstacles_points(
            self.num_of_hard_obstacles,
            min_dim=200,
            max_dim=400,
        )
        self._generate_obstacles_points(
            self.num_of_medium_obstacles,
            min_dim=150,
            max_dim=300,
        )
        self._generate_obstacles_points(
            self.num_of_small_obstacles,
            min_dim=50,
            max_dim=150,
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
                    new_obstacle, check_target="agent"
                ) or self.robot.is_overlapped(new_obstacle, check_target="goal")
            self.obstacles += new_obstacle

    def _get_viewer(self):
        return self.viewer
