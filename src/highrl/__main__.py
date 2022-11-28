"""Train your robot with a teacher to guide the process

Usage
------------------
    
    $ highrl [--options]

Get data about the arguments

    $ highrl -help

Or just run the following to the default
    
    $ highrl

Do not forget to specify the output dir for models saving

    $ highrl --output_dir=<your-ouput-dir>
    
    
Version
------------------
 - highrl v0.0.5
"""


import os
from highrl.envs.teacher_env import TeacherEnv
from time import time
from highrl.policy.policy_networks import LinearActorCriticPolicy
from highrl.policy.feature_extractors import LSTMFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
from highrl.utils.parser import parse_args, generate_agents_config, handle_output_dir
from highrl.callbacks.teacher_callback import TeacherLogCallback


def main() -> None:
    args = parse_args()
    print(f">>> Start training {args.env_mode} ... ")

    if args.env_mode == "teacher":
        robot_config, teacher_config = generate_agents_config(
            args.robot_config_path, args.teacher_config_path
        )

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

        args = handle_output_dir(args=args)
        planner_env = TeacherEnv(
            robot_config=robot_config,
            teacher_config=teacher_config,
            robot_output_dir=args.robot_models_path,
        )
        policy_kwargs = {
            "features_extractor_class": LSTMFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=5),
        }
        logpath = os.path.join(args.teacher_logs_path, "teacher_logs.csv")
        model = PPO(LinearActorCriticPolicy, planner_env, verbose=1 , policy_kwargs = policy_kwargs)
        callback = TeacherLogCallback(train_env = planner_env, logpath=logpath, eval_freq=1, verbose=0)
        model.learn(total_timesteps=int(1e7),  callback=callback)
        model.save(f"{args.teacher_models_dir}/model_{int(time())}")
    else:
        raise NotImplementedError("robot training is not implemented")


if __name__ == "__main__":
    main()
