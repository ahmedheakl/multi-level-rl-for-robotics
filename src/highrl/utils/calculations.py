"""Impelementation of functions for mathematicals calculations"""
from typing import Tuple, List, Union
import numpy as np


def calculate_norm(point: Tuple[int, int]) -> float:
    """Calculate norm of a vector
        v = (x, y)
        |v| = sqrt(x**2 + y**2)
    """
    x_value, y_value = point
    distance = (x_value**2 + y_value**2) ** 0.5
    return distance


def difference_vectors(
    first_vector: Tuple[int, int],
    second_vector: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Calculate the difference between two vectors (s1 - s2)
    """
    vector1_x, vector1_y = first_vector
    vector2_x, vector2_y = second_vector
    return (vector1_x - vector2_x, vector1_y - vector2_y)


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


def point_to_point_distance(
    first_point: Tuple[int, int],
    second_point: Tuple[int, int],
) -> float:
    """Get distance between two points

    Args:
        first_point (Tuple[int, int]): First points coordinates
        second_point (Tuple[int, int]): Second points coordinates

    Returns:
        float: Distance between points
    """
    point1_x, point1_y = first_point
    point2_x, point2_y = second_point

    return ((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2) ** 0.5


def cross_product_point_line(
    point: List[int], p_first: List[int], p_sec: List[int]
) -> Union[float, int]:
    """Calculate the cross product between a point and a line

    Args:
        point (List[int]): input point
        p_first (List[int]): first point of the line
        p_sec (List[int]): second point of the line

    Returns:
        float: value of the cross product
    """
    ground_point: np.ndarray = np.array(point, dtype=np.float32)
    line_point1: np.ndarray = np.array(p_first, dtype=np.float32)
    line_point2: np.ndarray = np.array(p_sec, dtype=np.float32)
    left_point = ground_point - line_point1
    right_point = ground_point - line_point2
    cross_product = left_point[0] * right_point[1] - left_point[1] * right_point[0]
    return cross_product


def cross_product_triangle(
    point: Tuple[int, int],
    p_first: Tuple[int, int],
    p_sec: Tuple[int, int],
) -> Union[float, int]:
    """Triagular cross product
    (p1 - p) x (p2 - p)
    Args:
        point (List[int]): input point
        p_first (List[int]): first point of the line
        p_sec (List[int]): second point of the line

    Returns:
        float: triangular cross product value
    """
    ground_point = np.array(point, dtype=np.float32)
    pnt1 = np.array(p_first, dtype=np.float32)
    pnt2 = np.array(p_sec, dtype=np.float32)
    return np.cross((pnt1 - ground_point), (pnt2 - ground_point)).tolist()
