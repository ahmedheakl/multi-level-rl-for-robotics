from highrl.utils.parser import parse_args, generate_agents_config, handle_output_dir
from highrl.envs.env_encoders import EvalEnv2DPlayer
from highrl.envs.env_player import EnvPlayer
from highrl.envs.eval_env import RobotEvalEnv


if __name__ == "__main__":
    args = parse_args()
    robot_config, eval_config, teacher_config = generate_agents_config(
        args.robot_config_path, args.robot_eval_config_path, args.teacher_config_path
    )
    # env = RobotEvalEnv(robot_config, args)
    env = RobotEvalEnv(eval_config, args)
    player = EnvPlayer(env)
