"""Implementation of Teacher Environment"""
from typing import List, Tuple
import argparse
import logging
from random import uniform
import pandas as pd
from configparser import RawConfigParser
from gym import Env, spaces
import numpy as np
from prettytable import PrettyTable

from highrl.envs import env_encoders as env_enc
from highrl.utils.general import configure_teacher
from highrl.utils import training_utils as train_utils
from highrl.utils import teacher_utils as teach_utils

_LOG = logging.getLogger(__name__)


class TeacherEnv(Env):
    """Environment for training the teacher agent"""

    infinite_difficulty: int = 1080 * 720  # w * h
    tensorboard_dir: str = "runs/teacher"
    rob_avg_rwrd_grph_name: str = "robot_avg_reward"
    rob_avg_eps_steps_grph_name: str = "robot_avg_episode_steps"
    rob_suc_rate_grph_name: str = "robot_success_rate"
    techr_rwrd_grph_name: str = "teacher_reward"
    rob_num_suc_grph_name: str = "robot_num_successes"
    rob_lvl_grph_name: str = "robot_level"
    action_space_names: List[str] = ["robot_x", "robot_y", "goal_x", "goal_y"]

    def __init__(
        self,
        robot_config: RawConfigParser,
        eval_config: RawConfigParser,
        teacher_config: RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Box(
            low=0.01,
            high=0.99,
            shape=(8,),
            dtype=np.float32,
        )
        # [time_steps, robot_level, robot_reward, difficulty_area, difficulty_obs]
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=(6,),
            dtype=np.float32,
        )
        self.args = args
        self.done: bool = False

        self.cfg = configure_teacher(teacher_config)
        self.robot_metrics = train_utils.RobotMetrics()

        self._init_robot_env(robot_config, eval_config)
        self.opt.set_tb_writer(self.tensorboard_dir)

    def _init_robot_env(
        self,
        robot_config: RawConfigParser,
        eval_config: RawConfigParser,
    ) -> None:
        """Initialize robot environment"""
        if self.cfg.lidar_mode not in ["flat", "rings"]:
            raise ValueError(f"Lidar mode {self.cfg.lidar_mode} is not avaliable")

        if self.cfg.lidar_mode == "flat":
            robot_env = env_enc.RobotEnv1DPlayer(config=robot_config, args=self.args)
            eval_env = env_enc.EvalEnv1DPlayer(config=eval_config, args=self.args)

        else:
            robot_env = env_enc.RobotEnv2DPlayer(config=robot_config, args=self.args)
            eval_env = env_enc.EvalEnv2DPlayer(config=eval_config, args=self.args)

        self.opt = train_utils.TeacherMetrics(
            robot_env=robot_env,
            eval_env=eval_env,
            desired_difficulty=self.cfg.base_difficulty,
        )

    def _get_robot_metrics(self) -> None:
        """Calculates and prints training session results indicating how well the robot performed
        during this session. This is done for every robot trainig session created by the teacher.
        """
        if len(self.opt.results) <= 0:
            return

        total_reward: float = 0
        total_steps = 0
        num_success = 0
        # sample = [episode_reward, episode_steps, success_flag]
        for sample in self.opt.results:
            episode_reward, episode_steps, success_flag = sample
            total_reward += episode_reward
            total_steps += episode_steps
            num_success += success_flag
        self.robot_metrics.success_flag = num_success > 0
        len_results_str = f"\nlength of results: {len(self.opt.results)}"
        results_str = f"\nresults: {self.opt.results}"
        _LOG.debug(len_results_str)
        _LOG.debug(results_str)
        self.robot_metrics.avg_reward = total_reward / len(self.opt.results)
        self.robot_metrics.avg_episode_steps = total_steps / len(self.opt.results)
        self.robot_metrics.success_rate = num_success / len(self.opt.results)
        _LOG.info("Session %i Results", self.opt.time_steps)
        results_table = PrettyTable(
            field_names=["avg_reward", "avg_ep_steps", "success_rate"]
        )
        results_table.add_row(
            [
                f"{self.robot_metrics.avg_reward:0.2f}",
                f"{self.robot_metrics.avg_episode_steps:0.2f}",
                f"{self.robot_metrics.success_rate:0.2f}",
            ]
        )

        self.opt.tb_writer.add_scalar(
            self.rob_avg_rwrd_grph_name,
            self.robot_metrics.avg_reward,
            self.opt.time_steps,
        )
        self.opt.tb_writer.add_scalar(
            self.rob_avg_eps_steps_grph_name,
            self.robot_metrics.avg_episode_steps,
            self.opt.time_steps,
        )
        self.opt.tb_writer.add_scalar(
            self.rob_suc_rate_grph_name,
            self.robot_metrics.success_rate,
            self.opt.time_steps,
        )
        _LOG.info(results_table)

    def render(self, mode):
        """Idle render"""
        return None

    def collect_stats(self) -> None:
        """Collect statistics and tensorboard data"""
        if not self.cfg.collect_statistics:
            return
        self.opt.session_statistics.loc[len(self.opt.session_statistics)] = [
            self.robot_metrics.iid,
            self.opt.reward,
            self.opt.robot_env.opt.episode_reward,
            self.opt.difficulty_area,
            self.opt.difficulty_obs,
            self.robot_metrics.level,
            self.opt.robot_env.opt.num_successes,
        ]

        self.opt.tb_writer.add_scalar(
            self.techr_rwrd_grph_name,
            self.opt.reward,
            self.opt.time_steps,
        )
        self.opt.tb_writer.add_scalar(
            self.rob_num_suc_grph_name,
            self.opt.robot_env.opt.num_successes,
            self.opt.robot_env.opt.total_steps,
        )
        self.opt.tb_writer.add_scalar(
            self.rob_lvl_grph_name,
            self.robot_metrics.level,
            self.opt.time_steps,
        )

    def step(self, action: List) -> Tuple:
        """Step into the new state using an action given by the teacher model

        Args:
            action (List): action to take

        Returns:
            Tuple: observation, reward, done, info
        """
        self.opt.time_steps += 1
        self.opt.reward = 0.0

        ######################### Initiate Robot Env ############################

        position_action = [action[0], action[1], action[2], action[3]]

        robot_pos, goal_pos = teach_utils.get_robot_position_from_action(
            position_action, self.opt, self.action_space_names
        )

        self.opt.robot_env.add_boarder_obstacles()

        obstacles = teach_utils.get_obstacles_from_action(action, self.opt, self.cfg)
        for obstacle in obstacles:
            self.opt.robot_env.obstacles.obstacles_list.append(obstacle)

        self.opt.robot_env.set_robot_position(robot_pos, goal_pos)
        self.opt.robot_env.opt.is_initial_state = True

        self.opt.robot_env.reset()

        (
            is_passed_inf_diff,
            is_goal_overlap_robot,
            is_goal_or_robot_overlap_obstacles,
        ) = teach_utils.compute_difficulty(self.opt, self.infinite_difficulty)
        if (
            is_passed_inf_diff
            or is_goal_overlap_robot
            or is_goal_or_robot_overlap_obstacles
        ):
            self.opt.reward = (
                self.cfg.infinite_difficulty_penality * is_passed_inf_diff
                + self.cfg.overlap_goal_penality * is_goal_overlap_robot
                + self.cfg.is_goal_or_robot_overlap_obstacles_penality
                * is_goal_or_robot_overlap_obstacles
            )
            self.done = True
            self.collect_stats()
            self.opt.residual_steps = self.opt.time_steps
            return self._make_obs(), self.opt.reward, self.done, {}

        self.opt.residual_steps = 0

        self.opt.desired_difficulty = (
            self.cfg.base_difficulty
            * (self.cfg.diff_increase_factor) ** self.opt.episodes
        )

        # Calculating statistics and rewards
        self._get_robot_metrics()
        self.opt.terminal_state_flag = self.robot_metrics.success_flag and (
            self.opt.difficulty_area >= self.opt.desired_difficulty
        )

        if self.opt.terminal_state_flag:
            self.done = True
            self.opt.episodes += 1

        train_utils.start_robot_session(
            args=self.args,
            cfg=self.cfg,
            robot_metrics=self.robot_metrics,
            opt=self.opt,
        )

        # Flag to advance to next level
        advance_flag = uniform(0, 1) <= self.cfg.advance_probability
        self.collect_stats()

        self.opt.reward = teach_utils.get_reward(self.opt, self.cfg, self.robot_metrics)

        self.robot_metrics.level = (
            self.robot_metrics.level + advance_flag
        ) * advance_flag
        self.opt.robot_env.opt.num_successes = 0

        return (
            self._make_obs(),
            self.opt.reward,
            self.done,
            {"episodes_count": self.opt.episodes},
        )

    def _make_obs(self) -> List:
        """Create observations
        Returns:
            List: observation vector
        """
        return [
            self.robot_metrics.level,
            self.robot_metrics.avg_reward,
            self.opt.difficulty_area,
            self.opt.difficulty_obs,
            self.robot_metrics.avg_episode_steps,
            self.robot_metrics.success_rate,
        ]

    def get_time_steps(self) -> int:
        """Getter for the time steps instance variable"""
        return self.opt.time_steps

    def reset(self) -> List:
        """Resets teacher state

        Returns:
            List: observation vector
        """
        self.opt.time_steps = self.opt.residual_steps
        self.done = False
        return self._make_obs()
