from typing import List, Tuple
from highrl.envs.robot_env import RobotEnv
from gym import Env, spaces
from highrl.obstacle.single_obstacle import SingleObstacle
import numpy as np
from highrl.utils.calculations import *
from highrl.policy.feature_extractors import Robot1DFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
from random import randint, random, uniform
from highrl.utils.planner_checker import convex_hull_difficulty
from highrl.callbacks.robot_callback import RobotMaxStepsCallback
from time import time
import configparser
from highrl.envs.env_encoders import (
    RobotEnv1DPlayer,
    RobotEnv2DPlayer,
    EvalEnv1DPlayer,
    EvalEnv2DPlayer,
)
from highrl.envs.eval_env import RobotEvalEnv
import pandas as pd
import math
from highrl.callbacks.robot_callback import (
    RobotMaxStepsCallback,
    RobotLogCallback,
    RobotEvalCallback,
)
from stable_baselines3.common.callbacks import CallbackList
from prettytable import PrettyTable
import argparse
from os import path
import torch as th

# TODO: scale the observation space to [0,1]


class TeacherEnv(Env):
    INF_DIFFICULTY = 921600  # w * h = 1280 * 720

    def __init__(
        self,
        robot_config: configparser.RawConfigParser,
        eval_config: configparser.RawConfigParser,
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
        # self.action_space = spaces.Box(
        #     low=0.01, high=0.99, shape=(7,), dtype=np.float32
        # )
        # self.action_space = Graph(node_space=spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32), edge_space=None)
        # self.action_space = spaces.Gra
        # [time_steps, robot_level, robot_reward, difficulty_area, difficulty_obs]
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(6,), dtype=np.float32
        )
        self.args = args
        self.episodes = 0
        self.difficulty_area = 0
        self.difficulty_obs = 0
        self.time_steps = 0
        self.robot_level = 0
        self.done = 0
        self.robot_success_flag = 0
        self.previous_save_path = ""

        self.terminal_state_flag = 0

        self.robot_avg_reward = 0
        self.robot_success_rate = 0
        self.robot_avg_episode_steps = 0
        self.robot_id = 0

        self._configure(config=teacher_config)
        self.session_statistics = None
        if self.collect_statistics:
            self.session_statistics = pd.DataFrame(
                columns=[
                    "robot_id",
                    "teacher_reward",
                    "robot_episode_reward",
                    "current_difficulty_area",
                    "current_difficulty_obst",
                    "robot_level",
                ]
            )
        # self.eval_env = RobotEvalEnv(config=eval_config, args=self.args)

        if self.lidar_mode == "flat":
            self.robot_env = RobotEnv1DPlayer(config=robot_config, args=self.args)
            self.eval_env = EvalEnv1DPlayer(config=eval_config, args=self.args)

        elif self.lidar_mode == "rings":
            self.robot_env = RobotEnv2DPlayer(config=robot_config, args=self.args)
            self.eval_env = EvalEnv2DPlayer(config=eval_config, args=self.args)
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
        self.gamma = config.getfloat("reward", "gamma")
        # start with base difficulty
        self.desired_difficulty = self.base_difficulty
        self.diff_increase_factor = config.getfloat("reward", "diff_increase_factor")

        self.advance_probability = config.getfloat("env", "advance_probability")
        self.max_hard_obstacles_count = config.getint("env", "max_hard_obstacles_count")
        self.max_medium_obstacles_count = config.getint(
            "env", "max_medium_obstacles_count"
        )
        self.max_small_obstacles_count = config.getint(
            "env", "max_small_obstacles_count"
        )
        self.hard_obstacles_min_dim = config.getint("env", "hard_obstacles_min_dim")
        self.hard_obstacles_max_dim = config.getint("env", "hard_obstacles_max_dim")
        self.medium_obstacles_min_dim = config.getint("env", "medium_obstacles_min_dim")
        self.medium_obstacles_max_dim = config.getint("env", "medium_obstacles_max_dim")
        self.small_obstacles_min_dim = config.getint("env", "small_obstacles_min_dim")
        self.small_obstacles_max_dim = config.getint("env", "small_obstacles_max_dim")
        self.lidar_mode = config.get("env", "lidar_mode")
        self.collect_statistics = config.getboolean("statistics", "collect_statistics")
        self.scenario = config.get("statistics", "scenario")
        self.robot_log_eval_freq = config.getint("statistics", "robot_log_eval_freq")
        self.n_robot_eval_episodes = config.getint(
            "statistics", "n_robot_eval_episodes"
        )
        self.render_eval = config.getboolean("render", "render_eval")
        
        # FIXME: edit
        max_num_obstacles = self.max_hard_obstacles_count +\
                            self.max_medium_obstacles_count +\
                            self.max_small_obstacles_count
                            
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(max_num_obstacles + 2, 4), dtype=np.float32)

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
            print(f"======== Session {self.time_steps-1} Results ========")
            results_table = PrettyTable(
                field_names=["avg_reward", "avg_ep_steps", "success_rate"]
            )
            results_table.add_row(
                [
                    f"{self.robot_avg_reward:0.2f}",
                    f"{self.robot_avg_episode_steps:0.2f}",
                    f"{self.robot_success_rate:0.2f}",
                ]
            )
            print(results_table)

    def step(self, action) -> Tuple:
        """Take a step in the environment
        Args:
            action (np.ndarray): action to take
        Returns:
            tuple: observation, reward, done, info
        """
        self.time_steps += 1
        self.reward = 0

        ######################### Initiate Robot Env ############################
        action = self._convert_action_to_dict_format(action)

        px, py, gx, gy = self._get_robot_position_from_action(action)

        self.robot_env.set_robot_position(px=px, py=py, gx=gx, gy=gy)
        self.robot_env.is_initial_state = True

        self.robot_env.add_boarder_obstacles()

        self._generate_obstacles_points(
            math.ceil(action["hard_obstacles_count"] * self.max_hard_obstacles_count),
            min_dim=self.hard_obstacles_min_dim,
            max_dim=self.hard_obstacles_max_dim,
        )
        self._generate_obstacles_points(
            math.ceil(
                action["medium_obstacles_count"] * self.max_medium_obstacles_count
            ),
            min_dim=self.medium_obstacles_min_dim,
            max_dim=self.medium_obstacles_max_dim,
        )
        self._generate_obstacles_points(
            math.ceil(action["small_obstacles_count"] * self.max_small_obstacles_count),
            min_dim=self.small_obstacles_min_dim,
            max_dim=self.small_obstacles_max_dim,
        )
        self.robot_env.reset()

        self.difficulty_area, self.difficulty_obs = convex_hull_difficulty(
            self.robot_env.obstacles,
            self.robot_env.robot,
            self.robot_env.width,
            self.robot_env.height,
        )
        is_passed_inf_diff = self.difficulty_area >= self.INF_DIFFICULTY
        is_goal_overlap_robot = self.robot_env.robot.is_robot_overlap_goal()
        if is_passed_inf_diff or is_goal_overlap_robot:
            self.reward = (
                self.infinite_difficulty_penality * is_passed_inf_diff
                + self.overlap_goal_penality * is_goal_overlap_robot
            )
            self.done = True
            if self.collect_statistics:
                self.session_statistics.loc[len(self.session_statistics)] = [  # type: ignore
                    self.robot_id,
                    self.reward,
                    self.robot_env.episode_reward,
                    self.difficulty_area,
                    self.difficulty_obs,
                    self.robot_level,
                ]
            return self._make_obs(), self.reward, self.done, {}

        too_close_to_goal_penality = (
            self.too_close_to_goal_penality
            * self.robot_env.robot.is_robot_close_to_goal(min_dist=1000)
        )

        self.desired_difficulty = (
            self.base_difficulty * ((self.diff_increase_factor) ** self.episodes)
        )

        policy_kwargs = dict(features_extractor_class=Robot1DFeatureExtractor)

        device = self.args.device_used

        if int(self.robot_level) == 0:
            self.robot_id += 1
            # fmt: off
            print("initiating model ...")
            model = PPO("MultiInputPolicy", self.robot_env, policy_kwargs=policy_kwargs, verbose=2, device=device)
        else:
            print("loading model ...")
            model = PPO.load(self.previous_save_path, self.robot_env, device=device)

        # fmt: off
        logpath = path.join(self.args.robot_logs_path, "robot_logs.csv")
        eval_logpath = path.join(self.args.robot_logs_path, "robot_eval_logs.csv")
        eval_model_save_path = path.join(self.args.robot_models_path, "test/best_tested_robot_model")
        log_callback =  RobotLogCallback(train_env = self.robot_env, logpath= logpath, eval_freq=self.robot_log_eval_freq, verbose=0)
        robot_callback = RobotMaxStepsCallback(max_steps=self.max_session_timesteps, verbose=0)
        eval_callback = RobotEvalCallback(eval_env =self.eval_env  ,
        n_eval_episodes=self.n_robot_eval_episodes,
        logpath=eval_logpath,
        savepath=eval_model_save_path,
        eval_freq=self.max_session_timesteps,
        verbose=1,
        render=self.render_eval,
    )
        callback = CallbackList([log_callback, robot_callback])
        
        model.learn(total_timesteps=int(1e9), reset_num_timesteps=False,
                    callback=callback)
        
        print("saving model ...")
        model_save_path = path.join(self.args.robot_models_path, f"train/model_{int(time())}_{self.robot_level}")
        self.previous_save_path = model_save_path
        model.save(model_save_path)


        ############################# Calculate Statistics and Rewards ##################################
        self._get_robot_metrics()
        self.terminal_state_flag = self.robot_success_flag and (self.difficulty_area >= self.desired_difficulty)
        self.reward = self.__get_reward() + too_close_to_goal_penality
        print(self.reward)

        if self.terminal_state_flag:
            self.done = True
            self.episodes += 1
        
        # Flag to advance to next level
        advance_flag = uniform(0, 1) <= self.advance_probability
         

        if self.collect_statistics:
            self.session_statistics.loc[len(self.session_statistics)] = [  # type: ignore
                self.robot_id,
                self.reward,
                self.robot_env.episode_reward,
                self.difficulty_area,
                self.difficulty_obs, 
                self.robot_level
            ]

        self.robot_level = (self.robot_level + advance_flag) * advance_flag
        return self._make_obs(), self.reward, self.done, {"episodes_count": self.episodes}

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
            self.difficulty_area,
            self.difficulty_obs,
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
        # FIXME: edit
        def conv(prob, max):
            prob = np.clip(prob, 0.1, 0.9)
            return np.round(prob * max)
        robot_pos = np.zeros((4), np.int32)
        # get robot/goal position
        robot_pos[0] = conv(action[0][0], self.robot_env.width)
        robot_pos[1] = conv(action[0][1], self.robot_env.height)
        robot_pos[2] = conv(action[0][2], self.robot_env.width)
        robot_pos[3] = conv(action[0][3], self.robot_env.height)
            
        # get obstacles count
        obs_count = np.zeros((3), np.int32)
        obs_count[0] = conv(action[1][0], self.max_hard_obstacles_count)
        obs_count[1] = conv(action[1][1], self.max_medium_obstacles_count)
        obs_count[2] = conv(action[1][2], self.max_small_obstacles_count)
        
        planner_output = {"robot_pos": robot_pos, "obs_count": obs_count}
        big_obstacles = np.zeros((obs_count[0], 4), np.uint32)
        med_obstacles = np.zeros((obs_count[1], 4), np.uint32)
        small_obstacles = np.zeros((obs_count[2], 4), np.uint32)
        
        width = self.robot_env.width
        height = self.robot_env.height
        dims_array = np.array([width, height, width, height], dtype=np.int32)
        offset = 2
        for i in range(obs_count[0].item()):
            x = np.reshape(conv(action[i + offset], dims_array), (-1, 4))
            print("XXXXX", x.shape, x, type(x))
            big_obstacles[i] = x
        planner_output["big_obs"] = big_obstacles
        
        offset += obs_count[0].item()
        for i in range(obs_count[1].item()):
            x = np.reshape(conv(action[i + offset], dims_array), (-1, 4))
            print("XXXXX", x.shape, x, type(x))
            med_obstacles[i] = x
        planner_output["med_obs"] = med_obstacles
        
        offset += obs_count[1].item()
        for i in range(obs_count[2].item()):
            small_obstacles[i] = np.reshape(conv(action[i + offset], dims_array), (-1, 4))
        planner_output["sm_obs"] = small_obstacles
        print(planner_output)
        

        # for i in range(len(action)):
        #     planner_output["{}".format(self.action_space_names[i])] = action[i]
        # print(f"======== Teacher action for Session {self.time_steps} ========")
        # names = ["px", "py", "gx", "gy", "h_cnt", "m_cnt", "s_cnt"]
        # action_table = PrettyTable()
        # for i, val in enumerate(list(planner_output.values())):
        #     action_table.add_column(fieldname=names[i], column=[val])
        # print(action_table)
        return planner_output

    def _get_robot_position_from_action(self, action: dict) -> Tuple:
        """Clip robot/ goal positions
        Args:
            action (dict): action dict from model
        Returns:
            Tuple: clipped positions
        """
        px = int(
            np.clip(
                self.robot_env.width * action["robot_X_position"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )  # type: ignore
        py = int(
            np.clip(
                self.robot_env.height * action["robot_Y_position"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )
        gx = int(
            np.clip(
                self.robot_env.width * action["goal_X_position"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )
        gy = int(
            np.clip(
                self.robot_env.height * action["goal_Y_position"],
                a_min=0,
                a_max=self.robot_env.height - 2,
            )
        )
        return px, py, gx, gy

    def exp(self, base, exponent):
        neg = base < 0.0
        ans = abs(base) ** exponent
        ans = ans * ((-1) ** neg)
        return ans
        
    def __get_reward(self) -> float:
        """Calculate current reward
        Returns:
            float: current reward
        """
        dfc_fact = self.difficulty_area / self.desired_difficulty
        rwd_fact = float(self.robot_avg_reward / self.max_robot_episode_reward)
        
        r_s = self.exp(dfc_fact*rwd_fact, self.alpha)
        
        
        r_t = (self.terminal_state_flag
                * (1 - self.robot_env.episode_steps / self.max_robot_episode_steps)
                * self.terminal_state_reward)
        
        r_d = (self.difficulty_area - self.desired_difficulty) * self.gamma
        print("XXXXXXXXXXXXXX REWARD DATA XXXXXXXXXXXXXXXX")
        print(dfc_fact, type(dfc_fact), rwd_fact, type(rwd_fact), self.alpha, type(self.alpha))
        
        reward = r_s + r_t + r_d
        
        return reward

    def _generate_obstacles_points(
        self, obstacles_count: int, min_dim: int, max_dim: int
    ) -> None:
        """Generate obstacles based on teacher action for next robot session
        Args:
            obstacles_count (int): number of obstacles
        """
        for i in range(int(obstacles_count)):
            overlap = True
            new_obstacle = SingleObstacle()
            while overlap:
                px = uniform(0, self.robot_env.width)
                py = uniform(0, self.robot_env.height)
                new_width = uniform(min_dim, max_dim)
                new_height = uniform(min_dim, max_dim)
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