"""Utilties implementation for training the teacher agent"""
from typing import Tuple, List
import math
import logging
from prettytable import PrettyTable

from highrl.utils.utils import TeacherConfigs, Position
from highrl.utils.training_utils import TeacherMetrics, RobotMetrics
from highrl.utils.teacher_checker import convex_hull_difficulty
from highrl.obstacle import SingleObstacle
from highrl.utils.calculations import neg_exp

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
        opt.difficulty_area, opt.difficulty_obs = convex_hull_difficulty(
            opt.robot_env.obstacles,
            opt.robot_env.robot,
            opt.width,
            opt.height,
        )
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
    opt: TeacherMetrics,
    cfg: TeacherConfigs,
) -> List[SingleObstacle]:
    """Convert action from index-format to function-format.
    The output of the planner is treated as a list of functions,
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
    x_func, y_func, w_func, h_func = action[4], action[5], action[6], action[7]
    x_max = opt.width**cfg.max_obstacles_count
    y_max = opt.height**cfg.max_obstacles_count

    x_func = math.ceil(x_func * x_max)
    w_func = math.ceil(w_func * x_max)
    y_func = math.ceil(y_func * y_max)
    h_func = math.ceil(h_func * y_max)

    obstacles = []
    for _ in range(cfg.max_obstacles_count):
        obs_x = x_func % opt.width
        obs_y = y_func % opt.height
        obs_w = min(w_func % opt.width, opt.width - obs_x)
        obs_h = min(h_func % opt.height, opt.height - obs_y)
        obstacles.append(SingleObstacle(obs_x, obs_y, obs_w, obs_h))

        x_func = x_func // opt.width
        w_func = w_func // opt.width
        y_func = y_func // opt.height
        h_func = h_func // opt.height
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
        action_val = min(max(action_val, 0.1), 0.9)
        planner_output[action_names[idx]] = action_val
        action_table.add_column(fieldname=names[idx], column=[action_val])

    _LOG.info("====== Teacher action for Session %i ========", opt.time_steps)
    _LOG.info(action_table)
    robot_pos = Position()
    goal_pos = Position()
    robot_pos.x_pos = min(int(opt.width * planner_output["robot_x"]), opt.width - 2)
    robot_pos.y_pos = min(int(opt.height * planner_output["robot_y"]), opt.height - 2)
    goal_pos.x_pos = min(int(opt.width * planner_output["goal_x"]), opt.width - 2)
    goal_pos.y_pos = min(int(opt.height * planner_output["goal_y"]), opt.height - 2)
    return robot_pos, goal_pos


def get_reward(
    opt: TeacherMetrics,
    cfg: TeacherConfigs,
    robot_metrics: RobotMetrics,
) -> float:
    """Calculate current reward

    Returns:
        float: current reward
    """
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
        * opt.robot_env.robot.is_robot_close_to_goal(min_dist=1000)
    )

    return reward + too_close_to_goal_penality
