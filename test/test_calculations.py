"""Testing calculations module"""
import unittest
import math

from highrl.utils import Position


class TestCalculations(unittest.TestCase):
    """Testing calculations"""

    def test_point_to_point(self) -> None:
        """Testing point to point distance"""
        tests = [
            [(1.0, 1.0), (2.0, 2.0), math.sqrt(2.0)],
            [(5.5, 7.0), (9.0, 1.0), 6.946221995],
        ]
        for test in tests:
            p_first = Position[float](test[0][0], test[0][1])
            p_sec = Position[float](test[1][0], test[1][1])
            value = p_first.distance(p_sec)
            expected = test[2]
            self.assertAlmostEqual(
                expected,
                value,
                msg=f"Expected: {expected:.3f}, Found: {value:.3f}",
            )
