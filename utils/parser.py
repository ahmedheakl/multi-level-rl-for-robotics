import argparse
from rich_argparse import RichHelpFormatter


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
        default="configs/robot_env.config",
        dest="robot_config_path",
        help="path of configuration file of robot environment",
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        default="configs/teacher_env.config",
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
        help="path to output results for robot mode",
    )
    parser.add_argument(
        "--teacher_output_dir",
        type=str,
        default="saved_models/teacher",
        help="path to output results for teacher mode",
    )

    parser.add_argument(
        "--lidar_mode",
        type=str,
        choices=["flat", "rings"],
        default="none",
        help="mode to process lidar flat=1D, rings=2D",
    )

    args = parser.parse_args()

    return args
