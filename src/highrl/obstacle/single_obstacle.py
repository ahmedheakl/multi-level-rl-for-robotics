"""Implementation for a single obstalce representation"""
from typing import List
import numpy as np


class SingleObstacle:
    """Representation of a single obstacle"""

    def __init__(
        self,
        px: int = 0,
        py: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self.px = px
        self.py = py
        self.width = width
        self.height = height

    def get_position(self) -> List[int]:
        """Getter for position coordinates"""
        return [self.px, self.py]

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

    def get_dimension(self) -> List[int]:
        """Getter for obstacle dimension"""
        return [self.width, self.height]

    def overlap(self, obstacle):
        """Check overlap with another obstacle"""
        raise NotImplementedError

    def get_grid_points(self) -> np.ndarray:
        """Get all the points inside an obstacle"""
        height, width = map(int, (self.height, self.width))
        points = [
            [self.px + x, self.py + y]
            for x in range(width + 1)
            for y in range(height + 1)
        ]
        points = np.array(points, dtype=np.int32)
        return points

    def __str__(self) -> str:
        return f"Obstacle: [{self.px}, {self.py}, {self.width}, {self.height}]"
