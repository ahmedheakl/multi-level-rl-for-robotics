"""Generating dataset points"""
from typing import Tuple, List
import random
import multiprocessing
import os
import argparse

import pandas as pd
import numpy as np

from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.agents.robot import Robot
from highrl.utils.teacher_checker import compute_difficulty
from highrl.utils import Position
from highrl.utils.logger import init_logger

HARD_OBS = 2
MED_OBS = 3
SMALL_OBS = 4
OBS_CNT = HARD_OBS + MED_OBS + SMALL_OBS
HARD_SIZE = 50
MED_SIZE = 40
SMALL_SIZE = 30
OUTPUT_SIZE = 4
ENV_SIZE = 256
NUM_POINTS: int = OUTPUT_SIZE * (OBS_CNT + 1)


def obstacle_level_gen(obs_cnt: int, obs_size: int) -> List[float]:
    """Random obstacles generator based on a certain count and size

    Note that generated obstacles are in the shape of rectangles.

    Args:
        obs_cnt (int): Number of obstacles to be generated
        obs_size (int): Maximum size of each generated obstacle

    Returns:
        List[SingleObstacle]: List of generated obstacles
    """
    obstacles: List[float] = []
    for _ in range(obs_cnt):
        obs_x = random.random() * ENV_SIZE
        obs_y = random.random() * ENV_SIZE
        obs_w = random.random() * obs_size
        obs_h = random.random() * obs_size

        obstacles.extend([obs_x, obs_y, obs_w, obs_h])

    return obstacles


def convert_from_points_to_objects(
    entries_points: List[float],
) -> Tuple[Robot, Obstacles]:
    """Convert from a list of floats representation to objects which are the robot and
    a list a of obstacles.
    """
    rob_x, rob_y, goal_x, goal_y = entries_points[:4]
    robot = Robot(Position[float](rob_x, rob_y), Position[float](goal_x, goal_y))
    obstacles_ls: List[SingleObstacle] = []
    for obs_j in range(4, NUM_POINTS, OUTPUT_SIZE):
        dims: List[int] = []
        for dim in range(OUTPUT_SIZE):
            dims.append(int(entries_points[obs_j + dim]))

        obstacles_ls.append(SingleObstacle(*dims))
    obstacles = Obstacles(obstacles_ls)
    return robot, obstacles


def random_env_generator() -> Tuple[float, List[float]]:
    """Random environment generator"""
    env_points = [random.uniform(0.0, 1.0) * ENV_SIZE for _ in range(4)]
    env_points.extend(obstacle_level_gen(HARD_OBS, HARD_SIZE))
    env_points.extend(obstacle_level_gen(MED_OBS, MED_SIZE))
    env_points.extend(obstacle_level_gen(SMALL_OBS, SMALL_SIZE))

    robot, obstacles = convert_from_points_to_objects(env_points)

    difficulty, _ = compute_difficulty(obstacles, robot, ENV_SIZE, ENV_SIZE)

    return difficulty, env_points


def generate_dataset(file_name: str, dataset_points: int) -> None:
    """Generate enviornments dataset.

    Input is difficulty value, label is a set of points representing
    the environment.
    record = [points, difficulty]
    """
    dataset_df = pd.DataFrame(
        np.zeros((dataset_points, 1 + NUM_POINTS), dtype=np.float32)
    )

    print(f"Generating {dataset_points} dataset points")
    for point_j in range(dataset_points):
        difficulty, env_points = random_env_generator()
        env_points.append(difficulty)
        dataset_df.loc[point_j] = env_points  # type: ignore

        if point_j > 0 and point_j % 100 == 0:
            print(f"Generated {point_j} points", point_j)

    dataset_df.to_csv(file_name)
    print(f"Succefully generated {dataset_points} points", dataset_points)


def generate_in_parallel(dataset_points: int, num_processes: int):
    """Generate dataset in parallel"""
    init_logger()
    dataset_collection = "env_gen"
    if not os.path.exists(dataset_collection):
        os.mkdir(dataset_collection)

    dataset_name = "env_points"
    dataset_points_per_process = int(dataset_points // num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for process_i in range(num_processes):
        process_dataset_name = f"{dataset_collection}/{dataset_name}{process_i}.csv"
        res = pool.apply_async(
            generate_dataset,
            (process_dataset_name, dataset_points_per_process),
        )
        results.append(res)

    pool.close()
    pool.join()
    data = [res.get() for res in results]
    return data


if __name__ == "__main__":
    num_processors = multiprocessing.cpu_count()
    print(f"You have {num_processors} processes available")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-points", type=int, default=10, dest="dataset_points")
    parser.add_argument(
        "--num-processes",
        choices=range(num_processors + 1),
        type=int,
        dest="num_processes",
        default=num_processors,
    )
    args = parser.parse_args()

    if args.num_processes > args.dataset_points:
        raise ValueError(
            "Number of processes cannot be larger than the number of dataset points"
        )

    generate_in_parallel(args.dataset_points, args.num_processes)
