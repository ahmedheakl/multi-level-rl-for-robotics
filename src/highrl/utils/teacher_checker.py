from highrl.obstacle.obstacles import Obstacles
from highrl.utils.calculations import (
    cross_product_point_line,
    cross_product_triangle,
    point_to_point_distance,
)
import numpy as np
from highrl.agents.robot import Robot
from typing import Tuple, List, Union, Callable


INF = 921600  # w * h = 1280 * 720
# EPSILON = 10 ** (-8)


def get_region_coordinates(
    harmonic_number: int, eps: int, coords: List[Union[int, float]]
) -> Tuple[List[List[float]], List[Callable[[float], List[float]]]]:
    """Calculates the boundaries for the current harmonic

    Args:
        harmonic_number (int): index of the desired harmonic
        eps (int): translation factor
        coords (List[Union[int, float]]): robot & goal coords [px, py, gx, py]

    Returns:
        Tuple[List[List[int]], List[Callable[[float], List[float]]]]: limiting coords & lines
    """
    px, py, gx, gy = coords
    slope = (gy - py) / ((gx - px))
    intercept = (gx * py - gy * px) / ((gx - px))
    shift_amount = eps * harmonic_number
    top_intercept = (slope * gy + gx) / (slope)
    bottom_intercept = (slope * py + px) / (slope)

    def left_line(x: float) -> List[float]:
        return [x, slope * x + (intercept + shift_amount)]

    def right_line(x: float) -> List[float]:
        return [x, slope * x + (intercept - shift_amount)]

    def bottom_line(x: float) -> List[float]:
        return [x, -1 / slope * x + bottom_intercept]

    def top_line(x: float) -> List[float]:
        return [x, -1 / slope * x + top_intercept]

    """
    p1-----p2
    |       |
    |       |
    p3-----p4
    """
    scale_factor = slope / (slope**2 + 1)
    x1: float = scale_factor * (top_intercept - (intercept + shift_amount))
    x2: float = scale_factor * (top_intercept - (intercept - shift_amount))
    x3: float = scale_factor * (bottom_intercept - (intercept - shift_amount))
    x4: float = scale_factor * (bottom_intercept - (intercept + shift_amount))
    x_coords = [x1, x2, x3, x4]
    points = [
        right_line(x_coords[i]) if ((i & 1) ^ (i >> 1)) else left_line(x_coords[i])
        for i in range(4)
    ]
    lines = [left_line, top_line, right_line, bottom_line]
    return points, lines


def check_if_point_inside_polygen(p: List[int], coords: List[List[float]]) -> bool:
    """Check if the input point is inside input polygen

    Args:
        p (List[int]): input point coordinates
        coords (List[List[int]]): polygen coordinates

    Returns:
        bool: flag whether the point is inside the polygen
    """
    lines_coords = {
        "top": [coords[0], coords[1]],
        "bottom": [coords[3], coords[2]],
        "left": [coords[3], coords[0]],
        "right": [coords[2], coords[1]],
    }
    # [top, bot, left, right]
    dirs = [cross_product_point_line(p, *line) for line in lines_coords.values()]
    eps = 1e-5
    vertical_check = (dirs[0] * dirs[1]) <= eps
    horizontal_check = (dirs[2] * dirs[3]) <= eps
    return vertical_check and horizontal_check


def check_valid_point(p: List[float], width: int, height: int) -> bool:
    """check if input point is within input constraints

    Args:
        p (List[int]): input point coordinates
        width (int): width constraint
        height (int): height constraint

    Returns:
        bool: flag whether point satisfies constraints
    """
    return p[0] >= 0 and p[1] >= 0 and p[0] <= width and p[1] <= height


def check_valid_path_existance(
    obstacles: Obstacles,
    coords: List[List[float]],
    width: int,
    height: int,
    robot_pos: List[int],
    goal_pos: List[int],
    omit_first_four: bool = True,
) -> bool:
    """Check if there is a valid path in the input segment

    Args:
        obstacles (Obstacles): obstacles object
        coords (List[List[int]]): coordinates defining the segment
        width (int): width of the env
        height (int): height of the env
        robot_pos (List[int]): robot position
        goal_pos (List[int]): goal position
        omit_first_four (bool): whether to ignore the first four obstacles which usually represent the boarder of the env

    Returns:
        bool: flag whether there exists a path
    """
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    num_dirs = 8
    # initialization
    px, py = robot_pos
    gx, gy = goal_pos
    env_map = []
    for x in range(width + 1):
        current = []
        for y in range(height + 1):
            current.append(".")
        env_map.append(current)

    for i, obstacle in enumerate(obstacles):
        if omit_first_four and i < 4:
            continue
        points = obstacle.get_grid_points()
        for p in points:
            if check_if_point_inside_polygen(p, coords):
                x, y = p
                env_map[x][y] = "X"
    # print(env_map)
    # bfs
    queue = [[px, py]]
    while len(queue) > 0:
        x, y = queue.pop(0)
        env_map[x][y] = "X"
        if x == gx and y == gy:
            return True
        for k in range(num_dirs):
            nx = x + dx[k]
            ny = y + dy[k]
            gen_point = [nx, ny]
            gen_point = np.array(gen_point, dtype=np.float32).tolist()
            if (
                check_if_point_inside_polygen(gen_point, coords)
                and check_valid_point(gen_point, width, height)
                and env_map[nx][ny] == "."
            ):
                env_map[nx][ny] = "X"
                queue.append([nx, ny])
    return False


def convex_hull_compute(points: List[List[int]]) -> List[List[int]]:
    """Compute convex hull polygen of input points

    Args:
        points (List[List[int]]): input points

    Returns:
        List[List[int]]: points defining convex hull polygen
    """
    points = sorted(points)
    convex_polygen = []
    for _ in range(2):
        sz = len(convex_polygen)
        for p in points:
            while len(convex_polygen) >= sz + 2:
                s, f = convex_polygen[-2:]
                if cross_product_triangle(p, f, s) <= 0:
                    break
                convex_polygen.pop(-1)
            convex_polygen.append(p)
        convex_polygen.pop(-1)
        points.reverse()
    return convex_polygen


def get_area_of_convex_polygen(points: List[List[int]]) -> float:
    """Compute the area of a polygen using its points

    Args:
        points (List[List[int]]): input points for the polygen

    Returns:
        float: area of input polygen
    """
    assert len(points), "Empty points list"
    num = len(points)
    points.append(points[0])
    array_points = np.array(points, dtype=np.float32)
    area = 0
    for i in range(num):
        area += np.cross(array_points[i], array_points[i + 1])
    area = abs(area)
    return area


def convex_hull_difficulty(
    obstacles: Obstacles, robot: Robot, width: int, height: int
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
    px, py = robot.get_position()
    gx, gy = robot.get_goal_position()
    eps = 5
    harmonic = 1
    while True:
        coords, _ = get_region_coordinates(harmonic, eps, [px, py, gx, gy])
        if check_valid_path_existance(
            obstacles, coords, width, height, [px, py], [gx, gy]
        ):
            points = [[px, py], [gx, gy]]
            num_overlap_obstacles = 0
            for obstacle in obstacles:
                overlapped = False
                obstacle_points = obstacle.get_grid_points()
                for p in obstacle_points:
                    if check_if_point_inside_polygen(p, coords):
                        points.append(p)
                        overlapped = True
                num_overlap_obstacles += overlapped
            convex_polygen = convex_hull_compute(points)
            if num_overlap_obstacles == 0:

                return point_to_point_distance((px, py), (gx, gy)), 0
            return get_area_of_convex_polygen(convex_polygen), num_overlap_obstacles
        max_x, max_y = np.max(np.array(coords), axis=0)
        min_x, min_y = np.min(np.array(coords), axis=0)
        harmonic += 1
        if max_x >= width and max_y >= height and min_x <= 0 and min_y <= 0:
            break
    return INF, max(0, len(obstacles.obstacles_list) - 4)
