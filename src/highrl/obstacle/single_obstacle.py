from typing import Tuple
import numpy as np


class SingleObstacle(object):
    def __init__(
        self, px: float = 0.0, py: float = 0.0, width: float = 0.0, height: float = 0.0
    ):
        self.px = px
        self.py = py
        self.width = width
        self.height = height

    def get_position(self) -> Tuple:
        return self.px, self.py

    def get_points(self):
        """
        p1----p2
        |     |
        |     |
        p4----p3
        """
        return [
            (self.px, self.py + self.height),
            (self.px + self.width, self.py + self.height),
            (self.px + self.width, self.py),
            (self.px, self.py),
        ]

    def get_dimension(self) -> Tuple:
        return self.width, self.height

    def overlap(self, obstacle):
        raise NotImplementedError

    def get_grid_points(self):
        h, w = map(int, (self.height, self.width))
        points = [
            [self.px + x, self.py + y] for x in range(w + 1) for y in range(h + 1)
        ]
        points = np.array(points, dtype=np.int32)
        return points

    def __str__(self) -> str:
        return f"Obstacle: [{self.px}, {self.py}, {self.width}, {self.height}]"
