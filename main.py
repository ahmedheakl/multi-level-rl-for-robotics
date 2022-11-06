from envs.teacher_env import TeacherEnv
from time import time
from policy.policy_networks import LinearActorCriticPolicy
from policy.feature_extractors import LSTMFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
import configparser
import argparse
from utils.parser import parse_args


def main(args: argparse.Namespace) -> None:

    print(f">>> Start training {args.env_mode} ... ")

    if args.env_mode == "teacher":
        robot_config = configparser.RawConfigParser()
        robot_config.read(args.robot_config_path)

        teacher_config = configparser.RawConfigParser()
        teacher_config.read(args.teacher_config_path)

        if args.render_each > -1:
            robot_config.set("render", "render_each", args.render_each)

        if args.lidar_mode != "none":
            teacher_config.set("env", "lidar_mode", str(args.lidar_mode))

        max_robot_steps = robot_config.getint("timesteps", "max_robot_steps")
        max_episode_timesteps = robot_config.getint("timesteps", "max_episode_steps")
        # fmt: off
        teacher_config.add_section("timesteps")
        teacher_config.set("timesteps", "max_robot_timesteps", str(max_robot_steps))
        teacher_config.set("timesteps", "max_episode_timesteps", str(max_episode_timesteps))

        planner_env = TeacherEnv(
            robot_config=robot_config,
            teacher_config=teacher_config,
            robot_output_dir=args.robot_output_dir,
        )
        policy_kwargs = {
            "features_extractor_class": LSTMFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=2),
        }
        # TODO: add LSTM to teacher
        model = PPO(LinearActorCriticPolicy, planner_env, verbose=1)
        model.learn(total_timesteps=int(1e7))
        model.save(f"{args.teacher_output_dir}/model_{int(time())}")
    else:
        raise NotImplementedError("robot training is not implemented")


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
