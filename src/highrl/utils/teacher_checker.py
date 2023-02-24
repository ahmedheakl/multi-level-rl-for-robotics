"""Difficulty computation implementation for the teacher"""
from typing import Tuple, List, Dict
import numpy as np

from highrl.obstacle.obstacles import Obstacles
from highrl.utils import Position
from highrl.agents.robot import Robot


INF = 256 * 256  # w * h


def get_region_coordinates(
    harmonic_number: int,
    eps: int,
    robot_goal_coords: List[int],
) -> List[Position]:
    """Calculates the boundaries for the current harmonic

    Args:
        harmonic_number (int): index of the desired harmonic
        eps (int): translation factor
        robot_goal_coords (List[int]): robot & goal coords [px, py, gx, py]

    Returns:
        List[List[int]]: limiting coords & lines
    """
    # Get the line coordinates connecting the goal and the robot
    robot_x, robot_y, goal_x, goal_y = robot_goal_coords
    slope = (goal_y - robot_y) / max(goal_x - robot_x, 1e-3)
    intercept = (goal_x * robot_y - goal_y * robot_x) / max(goal_x - robot_x, 1e-3)

    shift_amount = eps * harmonic_number

    # Top and bottom line intercepts with the y-axis
    top_intercept = (slope * goal_y + goal_x) / max(slope, 1e-4)
    bottom_intercept = (slope * robot_y + robot_x) / max(slope, 1e-4)

    def left_line(x_coords: float) -> Position[int]:
        y_pos = int(slope * x_coords + (intercept + shift_amount))
        return Position[int](int(x_coords), y_pos)

    def right_line(x_coords: float) -> Position[int]:
        y_pos = int(slope * x_coords + (intercept - shift_amount))
        return Position[int](int(x_coords), y_pos)

    # Coordinates of the points and lines (top_line, bottom ... etc)
    # Left Line: <p1, p3>
    # Right Line: <p2, p3>
    # Top Line: <p1, p2>
    # Bottom Line: <p3, p4>
    # p1-----p2
    # |       |
    # |       |
    # p3-----p4

    # Scale factor is just a part of the equations to calculate the coordinates
    # of the x-axis
    scale_factor = slope / (slope**2 + 1)
    top_left_x: float = scale_factor * (top_intercept - (intercept + shift_amount))
    top_right_x: float = scale_factor * (top_intercept - (intercept - shift_amount))
    botton_left_x: float = scale_factor * (
        bottom_intercept - (intercept - shift_amount)
    )
    botton_right_x: float = scale_factor * (
        bottom_intercept - (intercept + shift_amount)
    )
    x_coords = [top_left_x, top_right_x, botton_left_x, botton_right_x]

    # Compute point coordinates for defining the rectangle drawn above
    points = [
        right_line(x_coords[i]) if ((i & 1) ^ (i >> 1)) else left_line(x_coords[i])
        for i in range(4)
    ]
    return points


def is_point_inside_polygen(
    point: Position[int],
    coords: List[Position[int]],
) -> bool:
    """Check if the input point is inside input polygen

    Args:
        p (List[int]): input point coordinates
        coords (List[List[int]]): polygen coordinates

    Returns:
        bool: flag whether the point is inside the polygen
    """
    lines_coords: Dict[str, Tuple[Position[int], Position[int]]] = {
        "top": (coords[0], coords[1]),
        "bottom": (coords[3], coords[2]),
        "left": (coords[3], coords[0]),
        "right": (coords[2], coords[1]),
    }
    # [top, bot, left, right]
    dirs = [point.line_cross(*line) for line in lines_coords.values()]
    eps = 1e-5
    vertical_check = (dirs[0] * dirs[1]) <= eps
    horizontal_check = (dirs[2] * dirs[3]) <= eps
    return vertical_check and horizontal_check


def check_valid_point(point: np.ndarray, env_size: int) -> bool:
    """check if input point is within input constraints

    Note that that environment width and height are supposed to
    be equal.

    Args:
        p (np.ndarray): Input point coordinates
        env_size (int): Environment size

    Returns:
        bool: flag whether point satisfies constraints
    """
    is_valid = (point < env_size).all() and (point >= 0).all()
    return is_valid.item()


def if_there_is_a_path(
    obstacles: Obstacles,
    width: int,
    height: int,
    robot_pos: Position[int],
    goal_pos: Position[int],
) -> bool:
    """Check if the generated env from the teacher has a valid path"""
    delta_x = [-1, -1, -1, 0, 0, 1, 1, 1]
    delta_y = [-1, 0, 1, -1, 1, -1, 0, 1]
    num_dirs = 8
    # Intializing empty map
    # "." represents an empty space in the map
    env_map = []
    for _ in range(width + 1):
        current = []
        for _ in range(height + 1):
            current.append(".")
        env_map.append(current)

    # Filling obstacles spaces with "X" as means
    # of representing occupancy in the map
    for _, obstacle in enumerate(obstacles):
        points = obstacle.get_grid_points()
        for point in points:
            x_pos, y_pos = point
            if check_valid_point(point, width):
                env_map[x_pos][y_pos] = "X"
    # Breadth first search till you either find the
    # goal or you stop (aka. no valid in this map)
    # Intializing the queue with the robot position
    queue = [robot_pos]
    env_map[robot_pos.x][robot_pos.y] = "X"

    # Loop till you are out of non-visited points
    while len(queue) > 0:
        pos = queue.pop(0)

        # If goal is found, STOP
        if pos == goal_pos:
            return True

        # Loop over all directions
        for idx in range(num_dirs):
            new_pos = Position[int](pos.x + delta_x[idx], pos.y + delta_y[idx])
            if (
                check_valid_point(new_pos.get_coords(), width)
                and env_map[new_pos.x][new_pos.y] == "."
            ):
                # If the new point is inside the rectangle (between robot and goal),
                # it's within the boundaries of the env, and it's not occupied, add it
                # to the list of points to be explored.
                env_map[new_pos.x][new_pos.y] = "X"
                queue.append(new_pos)

    # If we reach here, the goal has not been reached
    return False


def check_valid_path_existance(
    obstacles: Obstacles,
    coords: List[Position[int]],
    width: int,
    height: int,
    robot_pos: Position[int],
    goal_pos: Position[int],
    omit_first_four: bool = True,
) -> bool:
    """Check if there is a valid path in the input segment

    Args:
        obstacles (Obstacles): Obstacles object
        coords (List[List[int]]): Coordinates defining the segment
        width (int): Width of the env
        height (int): Height of the env
        robot_pos (List[int]): Robot position
        goal_pos (List[int]): Goal position
        omit_first_four (bool): Whether to ignore the first four obstacles which usually
        represent the boarder of the env

    Returns:
        bool: flag whether there exists a path
    """
    delta_x = [-1, -1, -1, 0, 0, 1, 1, 1]
    delta_y = [-1, 0, 1, -1, 1, -1, 0, 1]
    num_dirs = 8

    # Intializing empty map
    # "." represents an empty space in the map
    env_map = []
    for _ in range(width + 1):
        current = []
        for _ in range(height + 1):
            current.append(".")
        env_map.append(current)

    # Filling obstacles spaces with "X" as means
    # of representing occupancy in the map
    for i, obstacle in enumerate(obstacles):
        if omit_first_four and i < 4:
            continue
        points = obstacle.get_grid_points()
        for point in points:
            pos = Position[int](point[0], point[1])
            if is_point_inside_polygen(pos, coords):
                x_pos, y_pos = point
                env_map[x_pos][y_pos] = "X"
    # Breadth first search till you either find the
    # goal or you stop (aka. no valid in this map)
    # Intializing the queue with the robot position
    queue = [robot_pos]
    env_map[robot_pos.x][robot_pos.y] = "X"

    # Loop till you are out of non-visited points
    while len(queue) > 0:
        pos = queue.pop(0)

        # If goal is found, STOP
        if pos == goal_pos:
            return True

        # Loop over all directions
        for idx in range(num_dirs):
            new_pos = Position[int](pos.x + delta_x[idx], pos.y + delta_y[idx])
            if (
                is_point_inside_polygen(new_pos, coords)
                and check_valid_point(new_pos.get_coords(), width)
                and env_map[new_pos.x][new_pos.y] == "."
            ):
                # If the new point is inside the rectangle (between robot and goal),
                # it's within the boundaries of the env, and it's not occupied, add it
                # to the list of points to be explored.
                env_map[new_pos.x][new_pos.y] = "X"
                queue.append(new_pos)

    # If we reach here, the goal has not been reached
    return False


def convex_hull_compute(points: List[Position[int]]) -> List[Position[int]]:
    """Compute convex hull polygen of input points

    Args:
        points (List[List[int]]): input points

    Returns:
        List[List[int]]: points defining convex hull polygen
    """
    points = sorted(points)
    convex_polygen: List[Position[int]] = []
    for _ in range(2):
        num_points = len(convex_polygen)
        for point in points:
            while len(convex_polygen) >= num_points + 2:
                second_point = convex_polygen[-2]
                first_point = convex_polygen[-1]
                if point.triangle_cross(first_point, second_point) <= 0:
                    break
                convex_polygen.pop(-1)
            convex_polygen.append(point)
        convex_polygen.pop(-1)
        points.reverse()
    return convex_polygen


def get_area_of_convex_polygen(points: List[Position[int]]) -> int:
    """Compute the area of a polygen using its points

    Args:
        points (List[List[int]]): Input points for the polygen

    Returns:
        float: Area of input polygen
    """
    assert len(points), "Empty points list"
    num = len(points)
    points.append(points[0])
    area = 0
    for i in range(num):
        area += points[i].inner_cross(points[i + 1])
    return abs(area)


def convex_hull_difficulty(
    obstacles: Obstacles,
    robot: Robot,
    width: int,
    height: int,
) -> Tuple[float, int]:
    """Calculate env complexity using convex_hull algorithm

    Args:
        obstacles (Obstacles): env obstacles
        robot (Robot): env robot
        width (int): env width
        height (int): env height

    Returns:
        Tuple[float, int]: env area difficulty, env obstacles difficulty
    """
    rob_pos = robot.get_position().to_int()
    goal_pos = robot.get_goal_position().to_int()
    eps = 1
    if if_there_is_a_path(
        obstacles,
        width,
        height,
        rob_pos,
        goal_pos,
    ):
        return INF, max(0, len(obstacles.obstacles_list) - 4)
    # Harmonic represents the number of expansions
    harmonic = 1
    while True:
        coords = get_region_coordinates(
            harmonic, eps, [rob_pos.x, rob_pos.y, goal_pos.x, goal_pos.y]
        )
        if check_valid_path_existance(
            obstacles, coords, width, height, rob_pos, goal_pos
        ):
            points: List[Position[int]] = [rob_pos, goal_pos]
            num_overlap_obstacles = 0
            for obstacle in obstacles:
                overlapped = False
                obstacle_points = obstacle.get_grid_points()
                for obs_pos in obstacle_points:
                    pos = Position[int](obs_pos[0], obs_pos[1])
                    if is_point_inside_polygen(pos, coords):
                        points.append(pos)
                        overlapped = True
                num_overlap_obstacles += overlapped
            convex_polygen = convex_hull_compute(points)
            if num_overlap_obstacles == 0:

                return robot.dist_to_goal(), 0
            return get_area_of_convex_polygen(convex_polygen), num_overlap_obstacles
        max_x = np.max(np.array(coords.x), axis=0)
        max_y = np.max(np.array(coords.y), axis=0)
        min_x = np.min(np.array(coords.x), axis=0)
        min_y = np.min(np.array(coords.y), axis=0)
        harmonic += 1
        if max_x <= 0 or max_y <= 0:
            break
        if max_x >= width and max_y >= height and min_x <= 0 and min_y <= 0:
            break
    return INF, max(0, len(obstacles.obstacles_list) - 4)
