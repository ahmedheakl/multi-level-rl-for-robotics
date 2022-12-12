from typing import List, Tuple
from gym import Env, spaces
from highrl.obstacle.single_obstacle import SingleObstacle
import numpy as np
from highrl.utils.calculations import *
from highrl.policy.feature_extractors import Robot1DFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
from random import uniform
from highrl.utils.teacher_checker import convex_hull_difficulty
from time import time
import configparser
from highrl.envs.env_encoders import (
    RobotEnv1DPlayer,
    RobotEnv2DPlayer,
    EvalEnv1DPlayer,
    EvalEnv2DPlayer,
)
import pandas as pd
import math
from highrl.callbacks.robot_callback import (
    RobotMaxStepsCallback,
    RobotLogCallback,
    RobotEvalCallback,
    RobotSuccessesCallback,
)
from stable_baselines3.common.callbacks import CallbackList
from prettytable import PrettyTable
import argparse
from os import path


class TeacherEnv(Env):

    INF_DIFFICULTY = 921600  # w * h = 1280 * 720

    def __init__(
        self,
        robot_config: configparser.RawConfigParser,
        eval_config: configparser.RawConfigParser,
        teacher_config: configparser.RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        """Construct the teacher enviornment

        Args:
            robot_config (configparser.RawConfigParser): robot env config object
            eval_config (configparser.RawConfigParser): eval env config object
            teacher_config (configparser.RawConfigParser): teacher env config object
            args (argparse.Namespace): args namespace with generated files output path

        Raises:
            ValueError: raises error if the user entered a lidar mode other than flat or rings
        """

        super(TeacherEnv, self).__init__()
        self.action_space_names = [
            "robot_x",
            "robot_y",
            "goal_x",
            "goal_y",
        ]
        self.action_space = spaces.Box(
            low=0.01, high=0.99, shape=(8,), dtype=np.float32
        )
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
        self.penality_time_step = 0

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
                    "robot_num_successes",  # robot num_successes in this teacher session
                ]
            )

        if self.lidar_mode == "flat":
            self.robot_env = RobotEnv1DPlayer(config=robot_config, args=self.args)
            self.eval_env = EvalEnv1DPlayer(config=eval_config, args=self.args)

        elif self.lidar_mode == "rings":
            self.robot_env = RobotEnv2DPlayer(config=robot_config, args=self.args)
            self.eval_env = EvalEnv2DPlayer(config=eval_config, args=self.args)
        else:
            raise ValueError(f"Lidar mode {self.lidar_mode} is not avaliable")

    def _configure(self, config: configparser.RawConfigParser) -> None:
        """Configure the environment variables using input config object
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
        self.is_goal_or_robot_overlap_obstacles_penality = config.getint(
            "reward", "is_goal_or_robot_overlap_obstacles_penality"
        )
        self.gamma = config.getfloat("reward", "gamma")
        # start with base difficulty
        self.desired_difficulty = self.base_difficulty
        self.diff_increase_factor = config.getfloat("reward", "diff_increase_factor")
        self.base_num_successes = config.getint("reward", "base_num_successes")
        self.num_successes_increase_factor = config.getfloat(
            "reward", "num_successes_increase_factor"
        )

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
        self.teacher_save_model_freq = config.getint("statistics", "save_model_freq")

        self.n_robot_eval_episodes = config.getint(
            "statistics", "n_robot_eval_episodes"
        )
        self.render_eval = config.getboolean("render", "render_eval")

    def _get_robot_metrics(self) -> None:
        """
        Calculates and prints training session results indicating how well the robot performed
        during this session. This is done for every robot trainig session created by the teacher.
        """
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
            print(f"======== Session {self.time_steps} Results ========")
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

    def step(self, action: List) -> Tuple:
        """Step into the new state using an action given by the teacher model

        Args:
            action (List): action to take

        Returns:
            Tuple: observation, reward, done, info
        """
        self.time_steps += 1
        self.reward = 0

        ######################### Initiate Robot Env ############################

        position_action = [action[0], action[1], action[2], action[3]]

        px, py, gx, gy = self._get_robot_position_from_action(position_action)

        self.robot_env.add_boarder_obstacles()

        obstacles = self.get_obstacles_from_action(action)
        for obstacle in obstacles:
            self.robot_env.obstacles.obstacles_list.append(obstacle)

        self.robot_env.set_robot_position(px=px, py=py, gx=gx, gy=gy)
        self.robot_env.is_initial_state = True

        self.robot_env.reset()

        self.difficulty_area, self.difficulty_obs = convex_hull_difficulty(
            self.robot_env.obstacles,
            self.robot_env.robot,
            self.robot_env.width,
            self.robot_env.height,
        )
        is_passed_inf_diff = self.difficulty_area >= self.INF_DIFFICULTY
        is_goal_overlap_robot = self.robot_env.robot.is_robot_overlap_goal()
        is_goal_or_robot_overlap_obstacles = 0
        for obstacle in self.robot_env.obstacles.obstacles_list:
            is_goal_or_robot_overlap_obstacles = (
                is_goal_or_robot_overlap_obstacles
                + self.robot_env.robot.is_overlapped(
                    obstacle=obstacle, check_target="agent"
                )
                + self.robot_env.robot.is_overlapped(
                    obstacle=obstacle, check_target="goal"
                )
            )
        is_goal_or_robot_overlap_obstacles = is_goal_or_robot_overlap_obstacles > 0
        print(
            f"is_goal_or_robot_overlap_obstacles = {is_goal_or_robot_overlap_obstacles}"
        )
        print(f"is_goal_overlap_robot = {is_goal_overlap_robot}")
        print(f"is_passed_inf_diff = {is_passed_inf_diff}")
        if (
            is_passed_inf_diff
            or is_goal_overlap_robot
            or is_goal_or_robot_overlap_obstacles
        ):
            self.reward = (
                self.infinite_difficulty_penality * is_passed_inf_diff
                + self.overlap_goal_penality * is_goal_overlap_robot
                + self.is_goal_or_robot_overlap_obstacles_penality
                * is_goal_or_robot_overlap_obstacles
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
                    self.robot_env.num_successes,
                ]
            self.penality_time_step = self.time_steps
            return self._make_obs(), self.reward, self.done, {}
        else:
            self.penality_time_step = 0

        too_close_to_goal_penality = (
            self.too_close_to_goal_penality
            * self.robot_env.robot.is_robot_close_to_goal(min_dist=1000)
        )

        self.desired_difficulty = (
            self.base_difficulty * (self.diff_increase_factor) ** self.episodes
        )

        self.num_successes = (
            self.base_num_successes
            * (self.num_successes_increase_factor) ** self.episodes
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
        robot_logpath = path.join(self.args.robot_logs_path, "robot_logs.csv")
        eval_logpath = path.join(self.args.robot_logs_path, "robot_eval_logs.csv")
        eval_model_save_path = path.join(self.args.robot_models_path, "test/best_tested_robot_model")
        log_callback =  RobotLogCallback(train_env = self.robot_env, logpath= robot_logpath, eval_frequency=self.robot_log_eval_freq, verbose=0)
        robot_max_steps_callback = RobotMaxStepsCallback(max_steps=self.max_session_timesteps, verbose=0)
        eval_callback = RobotEvalCallback(eval_env =self.eval_env  ,
        n_eval_episodes=self.n_robot_eval_episodes,
        logpath=eval_logpath,
        savepath=eval_model_save_path,
        eval_frequency=self.max_session_timesteps,
        verbose=1,
        render=self.render_eval,
    )
        successes_callback = RobotSuccessesCallback(num_successes=self.num_successes)
        callback = CallbackList([log_callback, robot_max_steps_callback, eval_callback, successes_callback])
        
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
                self.robot_level,
                self.robot_env.num_successes
            ]

        self.robot_level = (self.robot_level + advance_flag) * advance_flag
        self.robot_env.num_successes = 0
        return self._make_obs(), self.reward, self.done, {"episodes_count": self.episodes}

    def render(self):
        pass

    def _make_obs(self) -> List:
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

    def get_obstacles_from_action(self, action: np.array):
        """Convert action from index-format to function-format.
        The output of the planner is treated as a list of functions,
        each with valuable points in the domain from x -> [0, max_obs].

        The index is convert into the function by converting the base of the index
        from decimal(10) base to width/height.

        The function is then sampled at points x -> [0, max_obs] to obtain the corresponding values.
        There are 4 output values(indexes) from the teacher model:
            x position of obstacles
            y position of obstacles
            w width of obstacles
            h height of obstacles
        Each index is treated as a function and converted to obtain the corresponding values
        using the method described above.

        Args:
            action (np.array): input action(x, y, w, h)

        Returns:
            List[SingleObstalce, ...]: list of obtained obstacles
        """
        x_func, y_func, w_func, h_func = action[4], action[5], action[6], action[7]
        max_num_obstacles = (
            self.max_hard_obstacles_count
            + self.max_medium_obstacles_count
            + self.max_small_obstacles_count
        )
        width = self.robot_env.width
        height = self.robot_env.height
        x_max = width**max_num_obstacles
        y_max = height**max_num_obstacles

        x_func = math.ceil(x_func * x_max)
        w_func = math.ceil(w_func * x_max)
        y_func = math.ceil(y_func * y_max)
        h_func = math.ceil(h_func * y_max)

        obstacles = []
        for i in range(max_num_obstacles):
            x = x_func % width
            y = y_func % height
            w = min(w_func % width, width - x)
            h = min(h_func % height, height - y)
            obstacle = SingleObstacle(x, y, w, h)
            obstacles.append(obstacle)

            x_func = x_func // width
            w_func = w_func // width
            y_func = y_func // height
            h_func = h_func // height
        return obstacles

    def _get_robot_position_from_action(self, action: dict) -> Tuple:
        """Clip robot and goal positions to make sure they are inside the environment dimensions

        Args:
            action (dict): action dict from model

        Returns:
            Tuple: clipped positions of robot and goal
        """
        """Convert action form List format to dict format

        Args:
            action (List): output of teacher model

        Returns:
            dict: action dictionay (robot_x, robot_y, goal_x, goal_y, hard_obst_cnt, medium_obst_cnt, small_obst_cnt)
        """
        planner_output = {}
        action[0] = max(action[0], 0.099)
        action[0] = min(action[0], 0.899)
        action[1] = max(action[1], 0.099)
        action[1] = min(action[1], 0.899)
        action[2] = max(action[2], 0.1)
        action[2] = min(action[2], 0.9)
        action[3] = max(action[3], 0.1)
        action[3] = min(action[3], 0.9)

        for i in range(len(action)):
            planner_output["{}".format(self.action_space_names[i])] = action[i]
        print(f"======== Teacher action for Session {self.time_steps} ========")
        names = ["px", "py", "gx", "gy"]
        action_table = PrettyTable()
        for i, val in enumerate(list(planner_output.values())):
            action_table.add_column(fieldname=names[i], column=[val])
        print(action_table)

        px = int(
            np.clip(
                self.robot_env.width * planner_output["robot_x"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )  # type: ignore
        py = int(
            np.clip(
                self.robot_env.height * planner_output["robot_y"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )
        gx = int(
            np.clip(
                self.robot_env.width * planner_output["goal_x"],
                a_min=0,
                a_max=self.robot_env.width - 2,
            )
        )
        gy = int(
            np.clip(
                self.robot_env.height * planner_output["goal_y"],
                a_min=0,
                a_max=self.robot_env.height - 2,
            )
        )
        return px, py, gx, gy

    def __get_reward(self) -> float:
        """Calculate current reward
        Returns:
            float: current reward
        """
        dfc_fact = self.difficulty_area / self.desired_difficulty
        rwd_fact = float(self.robot_avg_reward / self.max_robot_episode_reward)

        r_s = self.exp(dfc_fact * rwd_fact, self.alpha)

        r_t = (
            self.terminal_state_flag
            * (1 - self.robot_env.episode_steps / self.max_robot_episode_steps)
            * self.terminal_state_reward
        )

        r_d = (self.difficulty_area - self.desired_difficulty) * self.gamma
        print(
            dfc_fact,
            type(dfc_fact),
            rwd_fact,
            type(rwd_fact),
            self.alpha,
            type(self.alpha),
        )

        reward = r_s + r_t + r_d

        return reward

    def exp(self, base, exponent):
        neg = base < 0.0
        ans = abs(base) ** exponent
        ans = ans * ((-1) ** neg)
        return ans

    def reset(self) -> List:
        """Resets teacher state

        Returns:
            List: observation vector
        """
        self.time_steps = self.penality_time_step
        self.done = 0
        return self._make_obs()
