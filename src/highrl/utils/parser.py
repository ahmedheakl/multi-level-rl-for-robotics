import argparse
from rich_argparse import RichHelpFormatter
from os import path, getcwd
from configparser import RawConfigParser
from highrl.configs import robot_config_str, teacher_config_str
from typing import Tuple


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
        "--robot_config",
        type=str,
        default="none",
        dest="robot_config_path",
        help="path of configuration file of robot environment",
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        default="none",
        dest="teacher_config_path",
        help="path of configuration file of teacher environment",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--env_mode",
        type=str,
        default="teacher",
        choices=["teacher", "robot"],
        help="which environment to use through training/testing",
    )

    parser.add_argument(
        "--render_each",
        type=int,
        default=-1,
        help="the frequency of rendering for robot environment",
    )
    parser.add_argument(
        "--robot_output_dir",
        type=str,
        default="saved_models/robot",
        help="relative path to output results for robot mode",
    )
    parser.add_argument(
        "--teacher_output_dir",
        type=str,
        default="saved_models/teacher",
        help="relative path to output results for teacher mode",
    )

    parser.add_argument(
        "--lidar_mode",
        type=str,
        choices=["flat", "rings"],
        default="flat",
        help="mode to process lidar flat=1D, rings=2D",
    )

    args = parser.parse_args()

    return args


def get_config(
    robot_config_path: str, teacher_config_path: str
) -> Tuple[RawConfigParser, RawConfigParser]:
    """Generates the robot and teacher configs

    Args:
        robot_config_path (str): path of the config file for robot
        teacher_config_path (str): path of the config file for teacher

    Returns:
        Tuple[RawConfigParser, RawConfigParser]: config files
    """
    robot_config = None
    teacher_config = None
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

    return (robot_config, teacher_config)
