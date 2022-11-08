from typing import Tuple


def calculate_norm(point):
    """
    Calculate norm of a vector
    v = (x, y)
    |v| = sqrt(x**2 + y**2)
    """
    x, y = point
    return (x**2 + y**2) ** 0.5


def difference_vectors(s1, s2):
    """
    Calculate the difference between two vectors (s1 - s2)
    """
    s1_x, s1_y = s1
    s2_x, s2_y = s2
    return (s1_x - s2_x, s1_y - s2_y)


def point_to_segment_distance(s1: Tuple, s2: Tuple, robot_position: Tuple):
    """Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    Args:
        s1 (Tuple): first point of the segment
        s2 (Tuple): second point of the segment
        robot_position (Tuple): robot coordinates

    Returns:
        float: closest distance between robot and segment
    """
    x1, y1 = s1
    x2, y2 = s2
    x3, y3 = robot_position
    base = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    left_side = ((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5
    right_side = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
    s = base + left_side + right_side
    area = (
        s * (s - base) * (s - left_side) * (s - right_side)
    ) ** 0.5  # area = 0.5 * height * base
    return 2.0 * area / base


def point_to_obstacle_distance(robot_position, obstalce_data):
    """
    Calculate the distance to an obstacle
    """
    px, py, height, width = obstalce_data
    segments = [
        [(px, py), (px + width, py)],
        [(px + width, py), (px + width, py + height)],
        [(px + width, py + height), (px, py + height)],
        [(px, py + height), (px, py)],
    ]

    distances = []

    for segment in segments:
        current_distance = point_to_segment_distance(
            segment[0], segment[1], robot_position
        )
        distances.append(current_distance)

    return distances


def point_to_point_distance(p1, p2):
    p1_x, p1_y = p1
    p2_x, p2_y = p2

    return ((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2) ** 0.5


if __name__ == "__main__":
    p1 = (1, 1)
    p2 = (1, 3)
    robo = (2, 4)
    print(point_to_segment_distance(p1, p2, robo))
