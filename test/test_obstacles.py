"""Tests for obstacles module"""
import unittest
import numpy as np

from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle


class ObstaclesTest(unittest.TestCase):
    """Testing obstacles creations and manipulations"""

    def test_flatten_contours_single_point(self) -> None:
        """Testing flattening the contours.

        Contours here are represented with the points representing an obstacle.
        For example, if there is an `SingleObstacle(px=0, py=0, w=10, h=10)`, then
        the points representing this obstacle are [(0, 0), (0, 10), (10, 0), (10, 10)].

        Now what we mean by flattening the cotours is instead of having the following form:
        `obstacles = [Obstacle(...), Obstacle(...), ...]`
        we have a flattened one:
        `contours = [(0, px0, py0), (0, px0+w, py0) ..., (1, px1, py1), ...]`
        """
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
        """Testing flattening the contours for multiple points"""
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
