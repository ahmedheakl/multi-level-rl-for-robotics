from envs.teacher_env import TeacherEnv
from time import time
from planner.planner import CustomActorCriticPolicy
from planner.planner import CustomLSTM
from stable_baselines3 import PPO  # type: ignore
import sys
import argparse


def main(args: argparse.Namespace) -> None:
    assert args.train_env_mode in [
        "teacher",
        "robot",
    ], f"input mode '{args.train_env_mode}' is not available"

    print(f">>>>>>>>>> Start training {args.train_env_mode} ... ")

    if args.train_env_mode == "teacher":
        planner_env = TeacherEnv()
        policy_kwargs = {
            "features_extractor_class": CustomLSTM,
            "features_extractor_kwargs": dict(features_dim=2),
        }
        model = PPO(CustomActorCriticPolicy, planner_env, verbose=1)
        model.learn(total_timesteps=int(1e7))
        model.save(f"saved_models/teacher/model_{int(time())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse arguments")
    parser.add_argument("--env_config", type=str, default="configs/robot_env.config")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_env_mode", type=str, default="teacher")
    parser.add_argument("--render_each", type=int, default=1)
    args = parser.parse_args()
    main(args=args)
