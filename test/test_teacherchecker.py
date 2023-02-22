"""Tests for teacher checker"""
from typing import List
import math
import unittest

from highrl.utils.teacher_checker import (
    get_region_coordinates,
    check_valid_path_existance,
    check_if_point_inside_polygen,
    convex_hull_compute,
)
from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle


class TeacherCheckerTest(unittest.TestCase):
    def test_region_coords_easy(self):
        value = sorted(
            get_region_coordinates(
                harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 5]
            )
        )
        expected = sorted([[3, 5], [4, 4], [1, 3], [2, 2]])
        for p, _ in enumerate(expected):
            value[p] = [round(value[p][i], 1) for i in range(len(value[p]))]
            self.assertListEqual(
                expected[p],
                value[p],
                msg=f"\nExpected:\n{expected[p]}\nFound:\n{value[p]}",
            )

    @classmethod
    def float_comparer(cls, a, b, msg=None):
        if len(a) != len(b):
            raise cls.failureException(msg)
        if not all(map(lambda args: math.isclose(*args), zip(a, b))):
            raise cls.failureException(msg)

    def test_region_coords_hard(self):
        value = get_region_coordinates(
            harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 7]
        )
        expected = [[3, 7], [4, 6], [2, 2], [1, 3]]
        self.addTypeEqualityFunc(list, self.float_comparer)
        for p, _ in enumerate(expected):
            value[p] = [round(value[p][i], 1) for i in range(len(value[p]))]
            self.assertEqual(
                expected[p],
                value[p],
                msg=f"\nExpected:\n{expected[p]}\nFound:\n{value[p]}",
            )

    def test_check_if_point_inside(self):
        coords = get_region_coordinates(
            harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 7]
        )
        points: List[List[int]] = [
            [3, 5],
            [2, 3],
            [4, 7],
            [3, 6],
            [4, 6],
            [5, 10],
            [2, 1],
        ]
        value = [check_if_point_inside_polygen(p, coords) for p in points]
        expected = [True] * 2 + [False] + [True] * 2 + [False] * 2
        self.assertListEqual(
            expected, value, msg=f"\nExpected:\n{expected}\nFound:\n{value}"
        )

    def test_valid_path_existance_false(self):
        coords = get_region_coordinates(
            harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 7]
        )

        obstacles = Obstacles([SingleObstacle(1, 4, 3, 1)])
        value = check_valid_path_existance(
            obstacles, coords, 6, 8, [2, 3], [4, 7], omit_first_four=False
        )
        expected = False
        self.assertEqual(
            expected, value, msg=f"\nExpected: {expected} || Found: {value}"
        )

    def test_convex_polygen_compute_easy(self):
        points: List[List[int]] = [[0, 0], [0, 2], [2, 2], [1, 1]]
        value = sorted(convex_hull_compute(points))
        expected = [[0, 0], [0, 2], [1, 1], [2, 2]]
        for i in range(len(points)):
            self.assertListEqual(
                expected[i], value[i], msg=f"\nExpected:\n{expected}\nFound:\n{value}"
            )

    def test_convex_polygen_compute_hard(self):
        points: List[List[int]] = [[2, 1], [2, 5], [3, 3], [4, 3], [4, 4], [6, 3]]
        value = convex_hull_compute(points)
        value.sort()
        expected: List[List[float]] = [[2, 1], [2, 5], [4, 4], [6, 3]]
        expected.sort()
        for i in range(len(value)):
            self.assertListEqual(
                expected[i], value[i], msg=f"\nExpected:\n{expected}\nFound:\n{value}"
            )
