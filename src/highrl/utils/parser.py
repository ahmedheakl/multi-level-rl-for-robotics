import argparse
from rich_argparse import RichHelpFormatter
from os import path, getcwd
import os
from configparser import RawConfigParser
from highrl.configs import robot_config_str, teacher_config_str, eval_config_str
from typing import Tuple
import getpass


def parse_args() -> argparse.Namespace:
    """Crease argument parser interface

    Returns:
        argparse.Namespace: namespace of input arguments
    """
    parser = argparse.ArgumentParser(
        prog="Parse arguments",
        description="Parse arguments to train teacher/robot environment",
        epilog="Enjoy the training! \N{slightly smiling face}",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--device-used-for-training",
        type=str,
        default="none",
        dest="device_used",
        choices=["CPU", "GPU"],
        help="what device to use for training (CPU/GPU)",
    )
    parser.add_argument(
        "--initial-teacher-model",
        type=str,
        default="none",
        dest="initial_teacher_model",
        help="path of initial teacher model used in training",
    )

    parser.add_argument(
        "--robot-config",
        type=str,
        default="none",
        dest="robot_config_path",
        help="path of configuration file of robot environment",
    )
    parser.add_argument(
        "--teacher-config",
        type=str,
        default="none",
        dest="teacher_config_path",
        help="path of configuration file of teacher environment",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="none",
        dest="eval_config_path",
        help="path of configuration file of teacher environment",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--env-mode",
        type=str,
        default="teacher",
        dest="env_mode",
        choices=["teacher", "robot"],
        help="which environment to use through training/testing",
    )

    parser.add_argument(
        "--render-each",
        type=int,
        default=-1,
        dest="render_each",
        help="the frequency of rendering for robot environment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="desktop",
        dest="output_dir",
        help="relative path to output results for robot mode",
    )
    parser.add_argument(
        "--lidar-mode",
        type=str,
        choices=["flat", "rings"],
        default="flat",
        dest="lidar_mode",
        help="mode to process lidar flat=1D, rings=2D",
    )

    args = parser.parse_args()

    return args


def generate_agents_config(
    robot_config_path: str, teacher_config_path: str, eval_config_path: str
) -> Tuple[RawConfigParser, RawConfigParser, RawConfigParser]:
    """Generates the robot and teacher configs

    Args:
        robot_config_path (str): path of the config file for robot env
        teacher_config_path (str): path of the config file for teacher env
        eval_config_path (str): path of the config file for eval env

    Returns:
        Tuple[RawConfigParser, RawConfigParser, RawConfigParser]: Tuple of config objects for
                                                                teacher, robot and eval envs
    """
    robot_config = None
    teacher_config = None
    eval_config = None
    if robot_config_path != "none":
        robot_config_path = path.join(getcwd(), robot_config_path)
        assert (
            path.exists(robot_config_path) == True
        ), f"path {robot_config_path} does not exist"
        robot_config = RawConfigParser()
        robot_config.read(robot_config_path)
    else:
        robot_config = RawConfigParser()
        robot_config.read_string(robot_config_str)

    if eval_config_path != "none":
        eval_config_path = path.join(getcwd(), eval_config_path)
        assert (
            path.exists(eval_config_path) == True
        ), f"path {eval_config_path} does not exist"
        eval_config = RawConfigParser()
        eval_config.read(eval_config_path)
    else:
        eval_config = RawConfigParser()
        eval_config.read_string(eval_config_str)

    if teacher_config_path != "none":
        teacher_config_path = path.join(getcwd(), teacher_config_path)
        assert (
            path.exists(teacher_config_path) == True
        ), f"path {teacher_config_path} does not exist"
        teacher_config = RawConfigParser()
        teacher_config.read(teacher_config_path)
    else:
        teacher_config = RawConfigParser()
        teacher_config.read_string(teacher_config_str)

    return (robot_config, teacher_config, eval_config)


def handle_output_dir(args: argparse.Namespace) -> argparse.Namespace:
    """Parse output dir from user and create output folders

    Args:
        args (argparse.Namespace): input args namespace.

    Returns:
        argparse.Namespace: args namespace with adjusted output path.
    """
    username = getpass.getuser()
    if args.output_dir == "desktop":
        args.output_dir = f"/home/{username}/Desktop"
    else:
        args.output_dir = path.join(getcwd(), args.output_dir)

    output_dir_path = path.join(args.output_dir, "output_dir")
    env_render_path = path.join(output_dir_path, "env_render")

    saved_models_path = path.join(output_dir_path, "saved_models")
    robot_models_path = path.join(saved_models_path, "robot")
    teacher_models_path = path.join(saved_models_path, "teacher")

    logs_path = path.join(output_dir_path, "logs")
    robot_logs_path = path.join(logs_path, "robot")
    teacher_logs_path = path.join(logs_path, "teacher")
    output_paths = {
        "output_dir_path": output_dir_path,
        "env_render_path": env_render_path,
        "saved_models_path": saved_models_path,
        "robot_models_path": robot_models_path,
        "teacher_models_path": teacher_models_path,
        "logs_path": logs_path,
        "robot_logs_path": robot_logs_path,
        "teacher_logs_path": teacher_logs_path,
    }

    for path_name, output_path in output_paths.items():
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        args.__setattr__(path_name, output_path)

    return args
