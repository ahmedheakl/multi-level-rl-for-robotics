"""Impelementation of functions for mathematicals calculations"""
from typing import Tuple, List


def point_to_segment_distance(
    robot_position: Tuple[int, int],
    segment_point1: Tuple[int, int],
    segment_point2: Tuple[int, int],
) -> float:
    """Calculate the closest distance between point(x3, y3) and a line segment with
    two endpoints (x1, y1), (x2, y2)

    Args:
        segment_point1 (Tuple): First point of the segment
        segment_point2 (Tuple): Second point of the segment
        robot_position (Tuple): Robot coordinates

    Returns:
        float: Closest distance between robot and segment
    """
    seg1_x, seg1_y = segment_point1
    seg2_x, seg2_y = segment_point2
    robot_x, robot_y = robot_position
    base = ((seg1_x - seg2_x) ** 2 + (seg1_y - seg2_y) ** 2) ** 0.5
    left_side = ((robot_x - seg1_x) ** 2 + (robot_y - seg1_y) ** 2) ** 0.5
    right_side = ((robot_x - seg2_x) ** 2 + (robot_y - seg2_y) ** 2) ** 0.5
    avg = (base + left_side + right_side) / 2
    area = (
        avg * (avg - base) * (avg - left_side) * (avg - right_side)
    ) ** 0.5  # area = 0.5 * height * base
    return 2.0 * area / base


def point_to_obstacle_distance(
    robot_position: Tuple[int, int],
    obstalce_data: Tuple[int, ...],
) -> List[float]:
    """Calculate the distance to an obstacle"""
    x_coord, y_coord, height, width = obstalce_data
    segments = [
        [(x_coord, y_coord), (x_coord + width, y_coord)],
        [(x_coord + width, y_coord), (x_coord + width, y_coord + height)],
        [(x_coord + width, y_coord + height), (x_coord, y_coord + height)],
        [(x_coord, y_coord + height), (x_coord, y_coord)],
    ]

    distances = []

    for segment in segments:
        current_distance = point_to_segment_distance(
            robot_position,
            *segment,
        )
        distances.append(current_distance)

    return distances


def neg_exp(base: float, exponent: float) -> float:
    """Exponentiation for negative numbers

    Since exponentiation of negative numbers might result
    in complex numbers, we omit this by returning sign * |base| ** exponent
    """
    neg = base < 0.0
    ans = abs(base) ** exponent
    ans = ans * ((-1) ** neg)
    return ans
