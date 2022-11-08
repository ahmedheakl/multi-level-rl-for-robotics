from typing import List, Tuple
from highrl.envs.robot_env import RobotEnv
from gym import Env, spaces
from highrl.obstacle.single_obstacle import SingleObstacle
import numpy as np
from highrl.utils.calculations import *
from highrl.policy.feature_extractors import Robot1DFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
from random import randint, random, uniform
from highrl.utils.planner_checker import PlannerChecker
from highrl.callbacks.robot_callback import RobotCallback
from time import time
import configparser
from highrl.envs.env_encoders import RobotEnv1DPlayer, RobotEnv2DPlayer

# TODO: scale the observation space to [0,1]


class TeacherEnv(Env):
    def __init__(
        self,
        robot_config: configparser.RawConfigParser,
        teacher_config: configparser.RawConfigParser,
        robot_output_dir: str,
    ) -> None:

        super(TeacherEnv, self).__init__()
        self.action_space_names = ["robot_position", "goal_position", "obstacles_count"]
        self.action_space = spaces.Box(
            low=0.01, high=0.99, shape=(3,), dtype=np.float32
        )
        # [time_steps, robot_level, robot_reward, current_difficulity]
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(5,), dtype=np.float32
        )

        self.robot_output_dir = robot_output_dir
        self.episodes = 0
        self.current_difficulty = 0
        self.time_steps = 0
        self.robot_level = 0
        self.done = 0
        self.current_robot_reward = 0
        self.robot_success_flag = 0
        self.previous_save_path = ""

        self.terminal_state_flag = 0  # gamma

        self.checker = PlannerChecker()

        self.robot_avg_reward = 0
        self.robot_success_rate = 0
        self.robot_avg_episode_steps = 0

        self._configure(config=teacher_config)

        if self.lidar_mode == "flat":
            self.robot_env = RobotEnv1DPlayer(config=robot_config)
        elif self.lidar_mode == "rings":
            self.robot_env = RobotEnv2DPlayer(config=robot_config)
        else:
            raise ValueError(f"Lidar mode {self.lidar_mode} is not avaliable")

    def _configure(self, config: configparser.RawConfigParser) -> None:
        """Configure the environment using input config file

        Args:
            config (configparser.RawConfigParser): input config object
        """
        self.config = config

        self.max_robot_episode_steps = config.getint(
            "timesteps", "max_episode_timesteps"
        )
        self.max_robot_timesteps = config.getint("timesteps", "max_robot_timesteps")

        self.alpha = config.getfloat("reward", "alpha")
        self.terminal_state_reward = config.getint("reward", "terminal_state_reward")
        self.max_robot_episode_reward = config.getint("reward", "max_reward")
        self.base_difficulty = config.getint("reward", "base_difficulty")
        # start with base difficulty
        self.desired_difficulty = self.base_difficulty

        self.advance_probability = config.getfloat("env", "advance_probability")
        self.max_obstacles_count = config.getint("env", "max_obstacles_count")
        self.lidar_mode = config.get("env", "lidar_mode")

    def _get_robot_metrics(self):
        if len(self.robot_env.results) > 0:
            total_reward = 0
            total_steps = 0
            num_success = 0
            # sample = [episode_reward, episode_steps, success_flag]
            for sample in self.robot_env.results:
                episode_reward, episode_steps, success_flag = sample
                total_reward += episode_reward
                total_steps += episode_steps
                num_success += success_flag
            self.robot_success_flag = num_success > 0
            self.robot_avg_reward = total_reward / len(self.robot_env.results)
            self.robot_avg_episode_steps = total_steps / len(self.robot_env.results)
            self.robot_success_rate = num_success / len(self.robot_env.results)
            print(f"|-----------------------------------------|")
            print(f"| avg_reward:   {self.robot_avg_reward:0.2f} |")
            print(f"| avg_ep_steps: {self.robot_avg_episode_steps:0.2f} |")
            print(f"| success_rate: {self.robot_success_rate:0.2f} |")
            print(f"|-----------------------------------------|")

    def step(self, action) -> Tuple:
        """Take a step in the environment

        Args:
            action (list): action to take

        Returns:
            tuple: observation, reward, done, info
        """
        self._get_robot_metrics()

        action = self._convert_action_to_dict_format(action)

        px, py, gx, gy = self._get_robot_position_from_action(action)

        self.robot_env.set_robot_position(px=px, py=py, gx=gx, gy=gy)
        self.robot_env.is_initial_state = True
        self.robot_env.reset()
        import math

        self._generate_obstacles_points(
            math.ceil(action["obstacles_count"] * self.max_obstacles_count)
        )

        args_list = list(map(int, [px, py, gx, gy]))
        self.current_difficulty = self.checker.get_map_difficulity(
            self.robot_env.obstacles,
            self.robot_env.width,
            self.robot_env.height,
            *args_list,
        )

        self.desired_difficulty = self.base_difficulty * (1.15) ** self.episodes

        policy_kwargs = dict(features_extractor_class=Robot1DFeatureExtractor)

        if self.robot_level == 0:
            # fmt: off
            model = PPO("MultiInputPolicy", self.robot_env, policy_kwargs=policy_kwargs, verbose=2)
        else:
            print("loading model ...")
            model = PPO.load(self.previous_save_path, self.robot_env)

        # fmt: off
        model.learn(total_timesteps=int(1e9), reset_num_timesteps=False,
                    callback=RobotCallback(max_steps=self.max_robot_timesteps, verbose=0))
        
        print("saving model ...")
        model_save_path = f"output_data/saved_models/robot/model_{int(time())}_{self.robot_level}"
        self.previous_save_path = model_save_path
        model.save(model_save_path)
        
        self.terminal_state_flag = self.robot_success_flag and (self.current_difficulty >= self.desired_difficulty)
        reward = self.__get_reward()

        if self.current_difficulty >= self.desired_difficulty:
            self.done = True
            self.episodes += 1

        self.time_steps += 1
        
        # Flag to advance to next level
        advance_flag = uniform(0, 1) <= self.advance_probability
        self.robot_level = (self.robot_level + advance_flag) * advance_flag

        return self._make_obs(), reward, self.done, {"episodes_count": self.episodes}

    def render(self):
        pass

    def _make_obs(self):
        """Create observations

        Returns:
            List: observation vector
        """
        return [
            self.robot_level,
            self.robot_avg_reward,
            self.current_difficulty,
            self.robot_avg_episode_steps,
            self.robot_success_rate,
        ]

    def _convert_action_to_dict_format(self, action):
        """Convert action form list format to dict format

        Args:
            action (list): output of planner model

        Returns:
            dict: action dictionay (robotPosition, goalPosition, numberOfObstacles)
        """
        # TODO: Find out why is the models stop being saved at Agent_Model_416
        planner_output = {}
        action[0] = max(action[0], 0.1)
        action[0] = min(action[0], 0.9)
        action[1] = max(action[1], 0.1)
        action[1] = min(action[1], 0.9)

        for i in range(len(action)):
            planner_output["{}".format(self.action_space_names[i])] = action[i]
        print("teacher_action = {}".format(planner_output))
        return planner_output

    def _get_robot_position_from_action(self, action: dict) -> Tuple:
        """Clip robot/ goal positions

        Args:
            action (dict): action dict from model

        Returns:
            Tuple: clipped positions
        """
        px = np.clip(
            self.robot_env.width * action["robot_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )  # type: ignore
        py = np.clip(
            self.robot_env.height * action["robot_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )
        gx = np.clip(
            self.robot_env.width * action["goal_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )
        gy = np.clip(
            self.robot_env.height * action["goal_position"],
            a_min=0,
            a_max=self.robot_env.height - 2,
        )
        return px, py, gx, gy

    def __get_reward(self) -> float:
        """Calculate current reward

        Returns:
            float: current reward
        """
        reward = (
            (self.current_difficulty / self.desired_difficulty)
            * (self.robot_avg_reward / self.max_robot_episode_reward)
        ) ** self.alpha + self.terminal_state_flag * (
            1 - self.time_steps / self.max_robot_episode_steps
        ) * self.terminal_state_reward
        return reward

    def _generate_obstacles_points(self, obstacles_count: int) -> None:
        """Generate obstacles based on teacher action for next robot session

        Args:
            obstacles_count (int): number of obstacles
        """
        self.robot_env.add_boarder_obstacles()
        for i in range(int(obstacles_count)):
            overlap = True
            new_obstacle = SingleObstacle()
            while overlap:
                px = randint(0, self.robot_env.width)
                py = randint(0, self.robot_env.height)
                new_width = randint(50, 500)
                new_height = randint(50, 500)
                new_obstacle = SingleObstacle(px, py, new_width, new_height)
                overlap = self.robot_env.robot.is_overlapped(new_obstacle)
            self.robot_env.obstacles += new_obstacle

    def reset(self):
        self.time_steps = 0
        self.done = 0
        return self._make_obs()
