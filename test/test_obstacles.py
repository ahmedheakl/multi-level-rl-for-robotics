"""Tests for obstacles module"""
import unittest
import numpy as np

from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle


class ObstaclesTest(unittest.TestCase):
    """Testing obstacles creations and manipulations"""

    def test_flatten_contours_single_point(self) -> None:
        """Testing flattening the contours"""
        obstacels_obj = Obstacles([SingleObstacle(0, 0, 10, 10)])
        value = obstacels_obj.get_flatten_contours()[0]
        expected = np.array(
            [[0, 0, 10], [0, 10, 10], [0, 10, 0], [0, 0, 0], [0, 0, 10]]
        )
        self.assertListEqual(
            expected.tolist(),
            value.tolist(),
            msg=f"Expected: {expected}, Found: {value}",
        )

    def test_flatten_contours_multiple_points(self) -> None:
        """Testing flattening the contours"""
        obstacels_obj = Obstacles(
            [SingleObstacle(0, 0, 10, 10), SingleObstacle(5, 10, 25, 30)]
        )
        value = obstacels_obj.get_flatten_contours()[0]
        expected = np.array(
            [
                [0, 0, 10],
                [0, 10, 10],
                [0, 10, 0],
                [0, 0, 0],
                [0, 0, 10],
                [1, 5, 40],
                [1, 30, 40],
                [1, 30, 10],
                [1, 5, 10],
                [1, 5, 40],
            ]
        )

        self.assertListEqual(
            expected.tolist(),
            value.tolist(),
            msg=f"Expected: {expected}, Found: {value}",
        )
