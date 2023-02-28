"""Tests for teacher checker"""
from typing import List
import time
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

    def test_execution_time(self) -> None:
        """Testing the time that the checker function has to run.
        Since this module is called > 400 time during the training,
        it should be as fast as possible. Hence, we need to make sure
        that this function does not take more than x-seconds per run."""

        max_expected_time = 25.0  # Measure in seconds
        env_size = 256
        obstacles = Obstacles(
            [SingleObstacle(30, 30, 100, 100), SingleObstacle(130, 130, 100, 100)]
        )
        robot = Robot(Position[float](0.2, 0.2), Position[float](253.8, 254.7))

        prev_time = time.time()
        diff, _ = compute_difficulty(obstacles, robot, env_size, env_size)
        total_time = time.time() - prev_time
        self.assertLess(
            diff,
            env_size * env_size,
            msg="Difficulty should not be maximum",
        )
        self.assertLess(
            total_time,
            max_expected_time,
            msg="Time limit exceeded in difficulty checker\n"
            + f"Expected {max_expected_time}, Found: {total_time:.2f}",
        )

    def test_execution_time_no_path(self) -> None:
        """Testing the time that the checker function has to run
        if the provided environment does not have a valid path."""
        max_expected_time = 25.0  # Measure in seconds
        env_size = 256
        obstacles = Obstacles([SingleObstacle(0, 0, env_size, env_size)])
        robot = Robot(Position[float](0.2, 0.2), Position[float](253.8, 254.7))

        prev_time = time.time()
        diff, _ = compute_difficulty(obstacles, robot, env_size, env_size)
        total_time = time.time() - prev_time
        self.assertEqual(
            diff,
            env_size * env_size,
            msg="Difficulty should be maximum",
        )
        self.assertLess(
            total_time,
            max_expected_time,
            msg="Time limit exceeded in difficulty checker\n"
            + f"Expected {max_expected_time}, Found: {total_time:.2f}",
        )
