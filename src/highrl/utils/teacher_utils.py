"""Utilties implementation for training the teacher agent"""
from typing import Tuple, List
import math
import time
import logging
from prettytable import PrettyTable
import torch as th
from stable_baselines3 import PPO  # type: ignore
from highrl.obstacle import Obstacles
from highrl.utils.general import TeacherConfigs
from highrl.utils.training_utils import TeacherMetrics, RobotMetrics
from highrl.utils.teacher_checker import compute_difficulty as convex_difficulty
from highrl.obstacle import SingleObstacle
from highrl.utils.calculations import neg_exp
from highrl.utils import Position

_LOG = logging.getLogger(__name__)


def compute_difficulty(
    opt: TeacherMetrics,
    infinite_difficulty: int,
) -> List[bool]:
    """Computing difficulty for the generated actions"""
    is_goal_overlap_robot = opt.robot_env.robot.is_robot_overlap_goal()
    is_goal_or_robot_overlap_obstacles = False
    for obstacle in opt.robot_env.obstacles.obstacles_list:
        is_goal_or_robot_overlap_obstacles = (
            is_goal_or_robot_overlap_obstacles
            + opt.robot_env.robot.is_overlapped(obstacle=obstacle, check_target="agent")
            + opt.robot_env.robot.is_overlapped(obstacle=obstacle, check_target="goal")
        )
    is_goal_or_robot_overlap_obstacles = is_goal_or_robot_overlap_obstacles > 0
    if is_goal_overlap_robot:
        opt.difficulty_area = 0
        opt.difficulty_obs = 0
        is_passed_inf_diff = False

    elif is_goal_or_robot_overlap_obstacles:
        opt.difficulty_area = infinite_difficulty
        opt.difficulty_obs = 1
        is_passed_inf_diff = True

    else:
        _LOG.warning("Computing difficulty")
        first_time = time.time()
        opt.difficulty_area, opt.difficulty_obs = convex_difficulty(
            opt.robot_env.obstacles,
            opt.robot_env.robot,
            opt.width,
            opt.height,
        )
        _LOG.warning("Computed difficulty in %i seconds", time.time() - first_time)
        is_passed_inf_diff = opt.difficulty_area >= infinite_difficulty

    _LOG.info("Goal/robot overlap obstacles: %s", is_goal_or_robot_overlap_obstacles)
    _LOG.info("Goal overlap robot: %s", is_goal_overlap_robot)
    _LOG.info("Passed inf diff: %s", is_passed_inf_diff)
    return [
        is_passed_inf_diff,
        is_goal_overlap_robot,
        is_goal_or_robot_overlap_obstacles,
    ]


def get_obstacles_from_action(
    action: List,
    cfg: TeacherConfigs,
) -> List[SingleObstacle]:
    """Convert action from index-format to function-format.
    The output of the teacher is treated as a list of functions,
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
        List[SingleObstalce: list of obtained obstacles
    """
    obstacles_ls = []

    # Convert obstacles position/dimension from [0, 1] to [0, width]
    max_big_obs = cfg.max_big_obstacles_count
    max_med_obs = cfg.max_med_obstacles_count
    max_small_obs = cfg.max_small_obstacles_count
    max_big_dim = cfg.big_obstacles_max_dim
    max_med_dim = cfg.med_obstacles_max_dim
    max_small_dim = cfg.small_obstacles_max_dim
    for idx in range(4, (4 * max_big_obs), 4):
        dims = [action[idx + dim_i] * max_big_dim for dim_i in range(4)]
        obstacles_ls.append(SingleObstacle(*dims))

    for idx in range((4 + 4 * max_big_obs), (4 * max_big_obs + 4 * max_med_obs), 4):
        dims = [action[idx + dim_i] * max_med_dim for dim_i in range(4)]
        obstacles_ls.append(SingleObstacle(*dims))

    for idx in range((4 + 4 * max_big_obs + 4 * max_med_obs), 40, 4):
        dims = [action[idx + dim_i] * max_small_dim for dim_i in range(4)]
        obstacles_ls.append(SingleObstacle(*dims))

    obstacles = Obstacles(obstacles_ls)
    return obstacles


def get_robot_position_from_action(
    action: List[float],
    opt: TeacherMetrics,
    action_names: List[str],
) -> Tuple[Position, Position]:
    """Clip robot and goal positions to make sure they are inside the environment dimensions

    Args:
        action (dict): action dict from model

    Returns:
        Tuple: clipped positions of robot and goal
    """
    planner_output = {}
    names = ["px", "py", "gx", "gy"]
    action_table = PrettyTable()
    for idx, action_val in enumerate(action):
        print(action_val, "\n")
        action_val = min(max(action_val, 0.1), 0.9)
        planner_output[action_names[idx]] = action_val
        action_table.add_column(fieldname=names[idx], column=[action_val])

    _LOG.info("====== Teacher action for Session %i ========", opt.time_steps)
    _LOG.info(action_table)
    robot_pos = Position[float](0.0, 0.0)
    goal_pos = Position[float](0.0, 0.0)
    robot_pos.x = min(int(opt.width * planner_output["robot_x"]), opt.width - 2)
    robot_pos.y = min(int(opt.height * planner_output["robot_y"]), opt.height - 2)
    goal_pos.x = min(int(opt.width * planner_output["goal_x"]), opt.width - 2)
    goal_pos.y = min(int(opt.height * planner_output["goal_y"]), opt.height - 2)
    return robot_pos, goal_pos


def get_reward(
    opt: TeacherMetrics,
    cfg: TeacherConfigs,
    robot_metrics: RobotMetrics,
) -> float:
    """Calculate teacher reward"""
    dfc_fact = opt.difficulty_area / opt.desired_difficulty
    rwd_fact = robot_metrics.avg_reward / cfg.max_robot_episode_reward

    r_s = neg_exp(dfc_fact * rwd_fact, cfg.alpha)

    r_t = (
        opt.terminal_state_flag
        * (1 - opt.robot_env.opt.episode_steps / cfg.max_robot_episode_steps)
        * cfg.terminal_state_reward
    )

    r_d = (opt.difficulty_area - opt.desired_difficulty) * cfg.gamma
    reward = r_s + r_t + r_d

    too_close_to_goal_penality = (
        cfg.too_close_to_goal_penality
        * opt.robot_env.robot.is_robot_close_to_goal(min_dist=15)
    )

    return reward + too_close_to_goal_penality


def load_env_generator(model_type: str, cfg):
    if model_type == "RL":
        model = PPO.load(cfg.generator_path)
        return model

    elif model_type == "SL":
        model = th.load(cfg.generator_path)
