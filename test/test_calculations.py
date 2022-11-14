import math
import unittest
from highrl.utils.calculations import *


class TestCalculations(unittest.TestCase):
    def test_norm_integers(self):
        value = calculate_norm(point=(3, 4))
        self.assertEqual(5, value, msg=f"Expected: {5}, found{value}")

    def test_norm_floats(self):
        value = calculate_norm(point=(3.5, 6.9))
        self.assertAlmostEqual(
            7.736924454,
            value,
            msg="Expected: {:.3f}, Found: {:.3f}".format(7.736924454, value),
        )

    def test_difference_vectors(self):
        value = difference_vectors((3.6, 4.2), (2.5, 8.9))
        expected = (1.1, -4.7)
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_point_to_point(self):
        tests = [
            [(1.0, 1.0), (2.0, 2.0), math.sqrt(2.0)],
            [(5.5, 7.0), (9.0, 1.0), 6.946221995],
        ]
        for test in tests:
            value = point_to_point_distance(test[0], test[1])
            expected = test[2]
            self.assertAlmostEqual(
                expected,
                value,
                msg="Expected: {:.3f}, Found: {:.3f}".format(expected, value),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
