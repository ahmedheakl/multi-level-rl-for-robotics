import numpy as np


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


def point_to_segment_distance(s1, s2, robot_position):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    x1, y1 = s1
    x2, y2 = s2
    x3, y3 = robot_position
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def point_to_obstacle_distance(robot_position, obstalce_data):
    """
    Calculate the distance to an obstacle
    """
    px, py, height, width = obstalce_data
    segments = [[(px, py), (px + width, py)],
                [(px + width, py),
                    (px + width, py + height)],
                [(px + width, py + height),
                    (px, py + height)],
                [(px, py + height), (px, py)]]

    distances = []

    for segment in segments:
        current_distance = point_to_segment_distance(
            segment[0], segment[1], robot_position)
        distances.append(current_distance)

    return distances


def point_to_point_distance(p1, p2):
    p1_x, p1_y = p1
    p2_x, p2_y = p2

    return ((p1_x - p2_x)**2 + (p1_y - p2_y)**2)**0.5


if __name__ == "__main__":
    p1 = (1, 1)
    p2 = (1, 3)
    robo = (2, 4)
    print(point_to_segment_distance(p1, p2, robo))
