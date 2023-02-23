"""Tests for teacher checker"""
from typing import List
import unittest

from highrl.utils.teacher_checker import (
    get_region_coordinates,
    check_valid_path_existance,
    check_if_point_inside_polygen,
    convex_hull_compute,
)
from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.utils import Position


class TeacherCheckerTest(unittest.TestCase):
    """Testing for teacher checker methods"""

    def test_region_coords_easy(self) -> None:
        """Testing computing the region coordinates"""
        value = sorted(
            get_region_coordinates(
                harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 5]
            )
        )
        expected = sorted([[3, 5], [4, 4], [1, 3], [2, 2]])
        for idx, expect in enumerate(expected):
            self.assertEqual(
                Position[int](expect[0], expect[1]),
                value[idx],
                msg=f"\nExpected:\n{expect}\nFound:\n{value[idx]}",
            )

    def test_region_coords_hard(self) -> None:
        """Testing computing the region coordinates"""
        value = get_region_coordinates(
            harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 7]
        )
        expected = [[3, 7], [4, 6], [2, 2], [1, 3]]
        for idx, expect in enumerate(expected):
            pos = Position[int](expect[0], expect[1])
            self.assertEqual(
                pos,
                value[idx],
                msg=f"\nExpected:\n{pos}\nFound:\n{value[idx]}",
            )

    def test_check_if_point_inside(self):
        """Testing for points inside a polygen checker"""
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
        value = [
            check_if_point_inside_polygen(Position[int](p[0], p[1]), coords)
            for p in points
        ]
        expected = [True] * 2 + [False] + [True] * 2 + [False] * 2
        self.assertListEqual(
            expected, value, msg=f"\nExpected:\n{expected}\nFound:\n{value}"
        )

    def test_valid_path_existance_false(self):
        """Testing for valid path checker"""
        coords = get_region_coordinates(
            harmonic_number=1, eps=1, robot_goal_coords=[2, 3, 4, 7]
        )

        obstacles = Obstacles([SingleObstacle(1, 4, 3, 1)])
        pos = Position[int](2, 3)
        gpos = Position[int](4, 7)
        value = check_valid_path_existance(
            obstacles, coords, 6, 8, pos, gpos, omit_first_four=False
        )
        expected = False
        self.assertEqual(
            expected, value, msg=f"\nExpected: {expected} || Found: {value}"
        )

    def test_convex_polygen_compute_easy(self):
        """Testing convex hull main problem"""
        points: List[List[int]] = [[0, 0], [0, 2], [2, 2], [1, 1]]
        pos_points = [Position(p[0], p[1]) for p in points]
        value = sorted(convex_hull_compute(pos_points))
        expected_points = [[0, 0], [0, 2], [1, 1], [2, 2]]
        expected = [Position[int](p[0], p[1]) for p in expected_points]
        for i in range(len(points)):
            self.assertEqual(
                expected[i], value[i], msg=f"\nExpected:\n{expected}\nFound:\n{value}"
            )

    def test_convex_polygen_compute_hard(self):
        """Testing convex hull main problem"""
        points: List[List[int]] = [[2, 1], [2, 5], [3, 3], [4, 3], [4, 4], [6, 3]]
        pos_points = [Position(p[0], p[1]) for p in points]
        value = convex_hull_compute(pos_points)
        value.sort()
        expected_points: List[List[int]] = [[2, 1], [2, 5], [4, 4], [6, 3]]
        expected = [Position[int](p[0], p[1]) for p in expected_points]
        expected.sort()
        for i, val in enumerate(value):
            self.assertEqual(
                expected[i], val, msg=f"\nExpected:\n{expected}\nFound:\n{value}"
            )
