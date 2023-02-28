"""Tests for teacher checker"""
from typing import List
import unittest

from highrl.utils.teacher_checker import (
    convex_hull_compute,
    get_path_bfs,
    compute_difficulty,
)
from highrl.utils import Position
from highrl.obstacle import SingleObstacle, Obstacles
from highrl.agents.robot import Robot


class TeacherCheckerTest(unittest.TestCase):
    """Testing for teacher checker methods"""

    def test_valid_path_existance_false(self):
        """Testing for valid path checker"""
        env_size = 6
        obstacles = Obstacles([SingleObstacle(1, 4, 3, 1)])
        rpos = Position[int](2, 3)
        gpos = Position[int](4, 7)
        value = get_path_bfs(
            obstacles=obstacles,
            env_size=env_size,
            robot_pos=rpos,
            goal_pos=gpos,
        )
        expected = False
        self.assertEqual(
            expected, value[0], msg=f"\nExpected: {expected} || Found: {value}"
        )

        self.assertListEqual([], value[1], msg=f"{value[1]} should be empty")

    def test_convex_polygen_compute_easy(self):
        """Testing convex hull main problem.

        Convex hull means that if we have a set of points, we want to
        construct the convex polygen with maximum area from these points.
        """
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

    def test_zero_div_in_slop_calc(self) -> None:
        """Testing for the zero division error in calculating the difficulty"""
        obstacles = Obstacles()
        rpos = Position[int](2, 2)
        gpos = Position[int](2, 3)
        robot = Robot(rpos, gpos)

        diff, _ = compute_difficulty(obstacles, robot, 5, 5)
        self.assertIsNot(diff, None)
