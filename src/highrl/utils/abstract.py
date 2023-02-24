"""Implementation for abstract classes that does not depend on any other modules"""
from typing import TypeVar, List, Generic
import numpy as np

# pylint: disable=invalid-name
TPosition = TypeVar("TPosition", bound="Position")

T = TypeVar("T", float, int)


class Position(Generic[T]):
    """Position of an agent"""

    def __init__(self, x_pos: T, y_pos: T) -> None:
        # pylint: disable=invalid-name
        self.x: T = x_pos
        self.y: T = y_pos

    def get_coords(self) -> np.ndarray:
        """Retrieve coords as numpy array"""
        return np.array([self.x, self.y])

    def set_pos(self, x_pos: T, y_pos: T) -> None:
        """Setter for position coordinates"""
        self.x = x_pos
        self.y = y_pos

    def __sub__(self: TPosition, other: TPosition):
        """Subtract two points"""
        x_pos = self.x - other.x
        y_pos = self.y - other.y
        return Position[T](x_pos, y_pos)

    def __add__(self: TPosition, other: TPosition):
        """Add two points"""
        x_pos = self.x + other.x
        y_pos = self.y + other.y
        return Position[T](x_pos, y_pos)

    def inner_cross(self: TPosition, other: TPosition) -> T:
        """Cross product with current point and other point"""
        return self.cross(self, other)

    @staticmethod
    def cross(current: TPosition, other: TPosition) -> T:
        """Cross products between two positions"""
        return np.cross(current.get_coords(), other.get_coords()).item()

    def triangle_cross(self, left_pos: TPosition, right_pos: TPosition) -> T:
        """Triangular cross product between current point and two other points"""
        left_diff = left_pos - self
        right_diff = right_pos - self
        return np.cross(left_diff.get_coords(), right_diff.get_coords()).item()

    def line_cross(self, left_pos: TPosition, right_pos: TPosition) -> T:
        """Cross product between a point and a line"""
        left_diff = self - left_pos
        right_diff = self - right_pos
        return self.cross(left_diff, right_diff)

    def distance(self: TPosition, other: TPosition) -> float:
        """Distance between current point and another point"""
        return np.linalg.norm(self.get_coords() - other.get_coords()).item()

    def to_list(self) -> List[T]:
        """Convert position to a list"""
        return [self.x, self.y]

    def __lt__(self: TPosition, other: TPosition) -> bool:
        """Overloading less than operator"""
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x

    def __le__(self: TPosition, other: TPosition) -> bool:
        """Overloading less than or equal operator"""
        if self.x == other.x:
            return self.y <= other.y
        return self.x <= other.x

    def __gt__(self: TPosition, other: TPosition) -> bool:
        """Overloading greater than operator"""
        if self.x == other.x:
            return self.y > other.y
        return self.x > other.x

    def __ge__(self: TPosition, other: TPosition) -> bool:
        """Overloading greater than or equal operator"""
        if self.x == other.x:
            return self.y >= other.y
        return self.x >= other.x

    def to_int(self):
        """Convert current position instance to int"""
        pos_x = int(self.x)
        pos_y = int(self.y)
        new_pos = Position[int](pos_x, pos_y)
        return new_pos

    def __eq__(self: TPosition, other: TPosition) -> bool:
        """Overloading equal operator"""
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        """Overloading string dunder method"""
        return f"Position(x={self.x}, y={self.y})"

    def __repr__(self) -> str:
        """Overloading string dunder method"""
        return f"Position(x={self.x}, y={self.y})"
