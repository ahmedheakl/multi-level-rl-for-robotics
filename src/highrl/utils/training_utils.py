"""Implementation of helper methods for training teacher and robot agents"""
from typing import Union
from dataclasses import dataclass
from argparse import Namespace
from os import path
from time import time
import pandas as pd
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import CallbackList
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from highrl.policy.feature_extractors import Robot1DFeatureExtractor
from highrl.callbacks import robot_callback
from highrl.utils.utils import TeacherConfigs, Position
from highrl.envs import env_encoders as env_enc


@dataclass
class RobotMetrics:
    """Metrics for the robot session"""

    success_flag: bool = False
    avg_reward: float = 0.0
    avg_episode_steps: float = 0.0
    success_rate: float = 0.0
    level: int = 0
    previous_save_path: str = ""
    iid: int = 0


@dataclass
class TeacherMetrics:
    """Metrics for teacher training"""

    robot_env: Union[env_enc.RobotEnv2DPlayer, env_enc.RobotEnv1DPlayer]
    eval_env: Union[env_enc.EvalEnv1DPlayer, env_enc.EvalEnv2DPlayer]
    tb_writer: SummaryWriter = SummaryWriter("runs")
    reward: float = 0.0
    episodes: int = 0
    difficulty_area: float = 0.0
    difficulty_obs: float = 0.0
    desired_difficulty: float = 0.0
    time_steps: int = 0
    terminal_state_flag: bool = False
    residual_steps: int = 0
    session_statistics: pd.DataFrame = pd.DataFrame(
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

    def set_tb_writer(self, tb_path: str) -> None:
        """Setter to tensorboard writter"""
        self.tb_writer = SummaryWriter(tb_path)

    @property
    def width(self) -> int:
        """Getter for environemnt width"""
        return self.robot_env.cfg.width

    @property
    def height(self) -> int:
        """Getter for environment height"""
        return self.robot_env.cfg.height

    @property
    def results(self):
        """Getter for robot results"""
        return self.robot_env.results


def start_robot_session(
    args: Namespace,
    cfg: TeacherConfigs,
    robot_metrics: RobotMetrics,
    opt: TeacherMetrics,
) -> None:
    """Start training the robot for a session"""
    policy_kwargs = {"features_extractor_class": Robot1DFeatureExtractor}

    if robot_metrics.level == 0:
        robot_metrics.iid += 1

        print("initiating model ...")
        model = PPO(
            "MultiInputPolicy",
            opt.robot_env,
            policy_kwargs=policy_kwargs,
            verbose=2,
            device=args.device,
        )
    else:
        print("loading model ...")
        model = PPO.load(
            robot_metrics.previous_save_path,
            opt.robot_env,
            device=args.device,
        )

    robot_logpath = path.join(args.robot_logs_path, "robot_logs.csv")
    eval_logpath = path.join(args.robot_logs_path, "robot_eval_logs.csv")
    eval_model_save_path = path.join(
        args.robot_models_path, "test/best_tested_robot_model"
    )
    log_callback = robot_callback.RobotLogCallback(
        train_env=opt.robot_env,
        logpath=robot_logpath,
        eval_frequency=cfg.robot_log_eval_freq,
        verbose=0,
    )
    robot_max_steps_callback = robot_callback.RobotMaxStepsCallback(
        max_steps=cfg.max_session_timesteps, verbose=0
    )
    eval_callback = robot_callback.RobotEvalCallback(
        eval_env=opt.eval_env,
        n_eval_episodes=cfg.n_robot_eval_episodes,
        logpath=eval_logpath,
        savepath=eval_model_save_path,
        eval_frequency=cfg.max_session_timesteps,
        verbose=1,
        render=cfg.render_eval,
    )
    successes_callback = robot_callback.RobotSuccessesCallback(
        num_successes=cfg.compute_success(opt.episodes)
    )
    callback = CallbackList(
        [log_callback, robot_max_steps_callback, eval_callback, successes_callback]
    )

    model.learn(total_timesteps=int(1e9), reset_num_timesteps=False, callback=callback)

    print("saving model ...")
    model_save_path = path.join(
        args.robot_models_path,
        f"train/model_{int(time())}_{robot_metrics.level}",
    )
    robot_metrics.previous_save_path = model_save_path
    model.save(model_save_path)
