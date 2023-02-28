"""Implementation for the evaluation environemnt"""
from random import randint
from argparse import Namespace
from configparser import RawConfigParser

from highrl.utils.abstract import Position
from highrl.envs.robot_env import RobotEnv
from highrl.obstacle.single_obstacle import SingleObstacle


class RobotEvalEnv(RobotEnv):
    """Evaluation environment for robot performance"""

    def __init__(self, config: RawConfigParser, args: Namespace):
        super(RobotEvalEnv, self).__init__(config, args)
        self.cfg.n_eval_episodes = config.getint("eval", "n_eval_episodes")
        self.add_boarder_obstacles()
        robot_init_x_pos = config.getint("eval", "robot_init_x_pos")
        robot_init_y_pos = config.getint("eval", "robot_init_y_pos")
        goal_x_pos = config.getint("eval", "goal_x_pos")
        goal_y_pos = config.getint("eval", "goal_y_pos")
        self.set_robot_position(
            Position[int](robot_init_x_pos, robot_init_y_pos),
            Position[int](goal_x_pos, goal_y_pos),
        )
        big_obs_count = config.getint("eval", "big_obs_count")
        med_obs_count = config.getint("eval", "med_obs_count")
        sml_obs_count = config.getint("eval", "sml_obs_count")

        # big_obs_pos = config.get("eval", "big_obs_pos")
        # med_obs_pos = config.get("eval", "med_obs_pos")
        # sml_obs_pos = config.get("eval", "sml_obs_pos")

        big_obs_pos = [(180, 180), (125, 125)]
        med_obs_pos = [(125, 70), (70, 125), (200, 50), (150, 30)]
        sml_obs_pos = [(30, 10), (10, 50), (100, 80), (20, 170)]

        big_obs_dim = config.getint("eval", "big_obs_dim")
        med_obs_dim = config.getint("eval", "med_obs_dim")
        sml_obs_dim = config.getint("eval", "sml_obs_dim")

        obstacles = self.generate_eval_obstacles(
            big_obs_count,
            med_obs_count,
            sml_obs_count,
            big_obs_pos,
            med_obs_pos,
            sml_obs_pos,
            big_obs_dim,
            med_obs_dim,
            sml_obs_dim,
        )

        for obstacle in obstacles:
            self.obstacles.obstacles_list.append(obstacle)

    def generate_eval_obstacles(
        self,
        big_obs_count,
        med_obs_count,
        sml_obs_count,
        big_obs_pos,
        med_obs_pos,
        sml_obs_pos,
        big_obs_dim,
        med_obs_dim,
        sml_obs_dim,
    ):
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


# class RobotEvalEnv(RobotEnv):
#     def __init__(self, config: RawConfigParser, args: Namespace):
#         super(RobotEvalEnv, self).__init__(config, args)
#         self._configure(config)
#         self.obstacles_configure()

#     def _configure(self, config: RawConfigParser) -> None:
#         """Configure the environment using input config file

#         Args:
#             config (RawConfigParser): input config object
#         """
#         self.config = config
#         self.robot_initial_px = config.getint("positions", "robot_initial_px")
#         self.robot_initial_py = config.getint("positions", "robot_initial_py")
#         self.robot_goal_px = config.getint("positions", "robot_goal_px")
#         self.robot_goal_py = config.getint("positions", "robot_goal_py")
#         self.width = config.getint("dimensions", "width")
#         self.height = config.getint("dimensions", "height")
#         self.robot_radius = config.getint("dimensions", "robot_radius")
#         self.goal_radius = config.getint("dimensions", "goal_radius")

#         self.num_of_hard_obstacles = config.getint("obstacles", "n_hard")
#         self.num_of_medium_obstacles = config.getint("obstacles", "n_medium")
#         self.num_of_small_obstacles = config.getint("obstacles", "n_small")
#         self.delta_t = config.getfloat("timesteps", "delta_t")
#         self.max_episode_steps = config.getint("timesteps", "max_episode_steps")

#         self.n_angles = config.getint("lidar", "n_angles")
#         self.lidar_angle_increment = config.getfloat("lidar", "lidar_angle_increment")
#         self.lidar_min_angle = config.getfloat("lidar", "lidar_min_angle")
#         self.lidar_max_angle = config.getfloat("lidar", "lidar_max_angle")

#         self.collision_score = config.getint("reward", "collision_score")
#         self.reached_goal_score = config.getint("reward", "reached_goal_score")
#         self.minimum_velocity = config.getfloat("reward", "minimum_velocity")
#         self.minimum_distance = config.getfloat("reward", "minimum_distance")
#         self.maximum_distance = config.getfloat("reward", "maximum_distance")
#         self.velocity_std = config.getfloat("reward", "velocity_std")
#         self.alpha = config.getfloat("reward", "alpha")
#         self.progress_discount = config.getfloat("reward", "progress_discount")

#         self.render_each = config.getint("render", "render_each")
#         self.save_to_file = config.getboolean("render", "save_to_file")

#         self.epsilon = config.getint("env", "epsilon")
#         self.collect_statistics = config.getboolean("statistics", "collect_statistics")
#         self.scenario = config.get("statistics", "scenario")

# def obstacles_configure(self) -> None:
#     self._generate_obstacles_points(
#         self.num_of_hard_obstacles,
#         min_dim=10,
#         max_dim=10,
#     )
#     self._generate_obstacles_points(
#         self.num_of_medium_obstacles,
#         min_dim=10,
#         max_dim=10,
#     )
#     self._generate_obstacles_points(
#         self.num_of_small_obstacles,
#         min_dim=10,
#         max_dim=10,
#     )

# def _generate_obstacles_points(
#     self, obstacles_count: int, min_dim: int, max_dim: int
# ) -> None:
#     """Generate obstacles based on teacher action for next robot session

#     Args:
#         obstacles_count (int): number of obstacles
#     """
#     self.add_boarder_obstacles()
#     for i in range(int(obstacles_count)):
#         overlap = True
#         new_obstacle = SingleObstacle()
#         while overlap:
#             px = randint(0, self.width)
#             py = randint(0, self.height)
#             new_width = randint(min_dim, max_dim)
#             new_height = randint(min_dim, max_dim)
#             new_obstacle = SingleObstacle(px, py, new_width, new_height)
#             overlap = self.robot.is_overlapped(
#                 new_obstacle, check_target="agent"
#             ) or self.robot.is_overlapped(new_obstacle, check_target="goal")
#         self.obstacles += new_obstacle

#     def _get_viewer(self):
#         return self.viewer
