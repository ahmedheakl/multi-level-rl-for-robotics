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
from highrl.callbacks.robot_callback import RobotMaxStepsCallback
from time import time
import configparser
from highrl.envs.env_encoders import RobotEnv1DPlayer, RobotEnv2DPlayer
import pandas as pd
import math
from highrl.callbacks.robot_callback import (
    RobotMaxStepsCallback,
    RobotLogCallback,
    RobotEvalCallback,
)
from stable_baselines3.common.callbacks import CallbackList
from os import path
import argparse

# TODO: scale the observation space to [0,1]


class TeacherEnv(Env):
    INF_DIFFICULTY = 2000

    def __init__(
        self,
        robot_config: configparser.RawConfigParser,
        teacher_config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:

        super(TeacherEnv, self).__init__()
        self.action_space_names = [
            "robot_X_position",
            "robot_Y_position",
            "goal_X_position",
            "goal_Y_position",
            "hard_obstacles_count",
            "medium_obstacles_count",
            "small_obstacles_count",
        ]
        self.action_space = spaces.Box(
            low=0.01, high=0.99, shape=(7,), dtype=np.float32
        )
        # [time_steps, robot_level, robot_reward, current_difficulity]
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(5,), dtype=np.float32
        )
        self.args = args
        self.episodes = 0
        self.current_difficulty = 0
        self.time_steps = 0
        self.robot_level = 0
        self.done = 0
        self.robot_success_flag = 0
        self.previous_save_path = ""

        self.terminal_state_flag = 0
        self.gamma = 1

        self.checker = PlannerChecker()

        self.robot_avg_reward = 0
        self.robot_success_rate = 0
        self.robot_avg_episode_steps = 0
        self.robot_id = 0

        self._configure(config=teacher_config)
        self.episode_statistics = None
        if self.collect_statistics:
            self.episode_statistics = pd.DataFrame(
                columns=[
                    "robot_id",
                    "reward",
                    "episode_reward",
                    "current_difficulty",
                    "robot_level",
                ]
            )

        if self.lidar_mode == "flat":
            self.robot_env = RobotEnv1DPlayer(config=robot_config, args=self.args)
        elif self.lidar_mode == "rings":
            self.robot_env = RobotEnv2DPlayer(config=robot_config, args=self.args)
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
        self.max_session_timesteps = config.getint("timesteps", "max_session_timesteps")

        self.alpha = config.getfloat("reward", "alpha")
        self.terminal_state_reward = config.getint("reward", "terminal_state_reward")
        self.max_robot_episode_reward = config.getint("reward", "max_reward")
        self.base_difficulty = config.getint("reward", "base_difficulty")
        self.overlap_goal_penality = config.getint("reward", "overlap_goal_penality")
        self.infinite_difficulty_penality = config.getint(
            "reward", "infinite_difficulty_penality"
        )
        self.too_close_to_goal_penality = config.getint(
            "reward", "too_close_to_goal_penality"
        )
        # start with base difficulty
        self.desired_difficulty = self.base_difficulty

        self.advance_probability = config.getfloat("env", "advance_probability")
        self.max_hard_obstacles_count = config.getint("env", "max_hard_obstacles_count")
        self.max_medium_obstacles_count = config.getint(
            "env", "max_medium_obstacles_count"
        )
        self.max_small_obstacles_count = config.getint(
            "env", "max_small_obstacles_count"
        )
        self.lidar_mode = config.get("env", "lidar_mode")
        self.collect_statistics = config.getboolean("statistics", "collect_statistics")
        self.scenario = config.get("statistics", "scenario")

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
            print(f"|------------------------------|")
            print(f"| avg_reward:   {self.robot_avg_reward:0.2f}  |")
            print(f"| avg_ep_steps: {self.robot_avg_episode_steps:0.2f} |")
            print(f"| success_rate: {self.robot_success_rate:0.2f}  |")
            print(f"|------------------------------|")

    def step(self, action) -> Tuple:
        """Take a step in the environment

        Args:
            action (list): action to take

        Returns:
            tuple: observation, reward, done, info
        """
        self.time_steps += 1
        self._get_robot_metrics()

        action = self._convert_action_to_dict_format(action)

        px, py, gx, gy = self._get_robot_position_from_action(action)

        self.robot_env.set_robot_position(px=px, py=py, gx=gx, gy=gy)
        self.robot_env.is_initial_state = True

        self._generate_obstacles_points(
            math.ceil(action["hard_obstacles_count"] * self.max_hard_obstacles_count),
            min_dim=300,
            max_dim=500,
        )
        self._generate_obstacles_points(
            math.ceil(
                action["medium_obstacles_count"] * self.max_medium_obstacles_count
            ),
            min_dim=100,
            max_dim=250,
        )
        self._generate_obstacles_points(
            math.ceil(action["small_obstacles_count"] * self.max_small_obstacles_count),
            min_dim=10,
            max_dim=50,
        )
        self.robot_env.reset()

        args_list = list(map(int, [px, py, gx, gy]))
        self.current_difficulty = self.checker.get_map_difficulity(
            self.robot_env.obstacles,
            self.robot_env.width,
            self.robot_env.height,
            *args_list,
        )
        if self.current_difficulty >= self.INF_DIFFICULTY:
            reward = self.infinite_difficulty_penality
            self.done = True
            return self._make_obs(), reward, self.done, {}

        if self.robot_env.robot.is_robot_overlap_goal():
            reward = self.overlap_goal_penality
            self.done = True
            return self._make_obs(), reward, self.done, {}

        if self.robot_env.robot.is_robot_close_to_goal(min_dist=1000):
            too_close_to_goal_penality = self.too_close_to_goal_penality
        else:
            too_close_to_goal_penality = 0

        self.desired_difficulty = self.base_difficulty * (1.15) ** self.episodes

        policy_kwargs = dict(features_extractor_class=Robot1DFeatureExtractor)

        if int(self.robot_level) == 0:
            self.robot_id += 1
            # fmt: off
            print("initiating model ...")
            model = PPO("MultiInputPolicy", self.robot_env, policy_kwargs=policy_kwargs, verbose=2)
        else:
            print("loading model ...")
            model = PPO.load(self.previous_save_path, self.robot_env)

        # fmt: off
        logpath = path.join(self.args.robot_logs_path, "robot_logs.csv")
        eval_logpath = path.join(self.args.robot_logs_path, "robot_eval_logs.csv")
        eval_model_save_path = path.join(self.args.robot_models_path, "test/best_tested_robot_model")
        log_callback =  RobotLogCallback(train_env = self.robot_env, logpath= logpath, eval_freq=100, verbose=0)
        robot_callback = RobotMaxStepsCallback(max_steps=self.max_session_timesteps, verbose=0)
        eval_callback = RobotEvalCallback(eval_env = self.robot_env,
        n_eval_episodes=10,
        logpath=eval_logpath,
        savepath=eval_model_save_path,
        eval_freq=self.max_session_timesteps,
        verbose=1,
        render=True,
    )
        callback = CallbackList([log_callback, robot_callback, eval_callback])
        model.learn(total_timesteps=int(1e9), reset_num_timesteps=False,
                    callback=callback)
        
        print("saving model ...")
        model_save_path = path.join(self.args.robot_models_path, f"train/model_{int(time())}_{self.robot_level}")
        self.previous_save_path = model_save_path
        model.save(model_save_path)
        
        self.terminal_state_flag = self.robot_success_flag and (self.current_difficulty >= self.desired_difficulty)
        reward = self.__get_reward() + too_close_to_goal_penality
        print(reward)

        if self.terminal_state_flag:
            self.done = True
            self.episodes += 1
        
        # Flag to advance to next level
        advance_flag = uniform(0, 1) <= self.advance_probability
        self.robot_level = (self.robot_level + advance_flag) * advance_flag

        if self.done:
            if self.collect_statistics:
                self.episode_statistics.loc[len(self.episode_statistics)] = [  # type: ignore
                    self.robot_id,
                    reward,
                    self.robot_env.episode_reward,
                    self.current_difficulty,
                    self.robot_level
                ]
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
            self.robot_env.width * action["robot_X_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )  # type: ignore
        py = np.clip(
            self.robot_env.height * action["robot_Y_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )
        gx = np.clip(
            self.robot_env.width * action["goal_X_position"],
            a_min=0,
            a_max=self.robot_env.width - 2,
        )
        gy = np.clip(
            self.robot_env.height * action["goal_Y_position"],
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
            (
                (self.current_difficulty / self.desired_difficulty)
                * (self.robot_avg_reward / self.max_robot_episode_reward)
            )
            ** self.alpha
            + (
                self.terminal_state_flag
                * (1 - self.robot_env.episode_steps / self.max_robot_episode_steps)
                * self.terminal_state_reward
            )
            + (self.current_difficulty - self.desired_difficulty) * self.gamma
        )
        return reward

    def _generate_obstacles_points(
        self, obstacles_count: int, min_dim: int, max_dim: int
    ) -> None:
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
                new_width = randint(min_dim, max_dim)
                new_height = randint(min_dim, max_dim)
                new_obstacle = SingleObstacle(px, py, new_width, new_height)
                overlap = self.robot_env.robot.is_overlapped(
                    new_obstacle, check_target="robot"
                ) or self.robot_env.robot.is_overlapped(
                    new_obstacle, check_target="goal"
                )
            self.robot_env.obstacles += new_obstacle

    def reset(self):
        self.time_steps = 0
        self.done = 0
        return self._make_obs()
