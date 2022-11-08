import unittest
from highrl.utils.planner_checker import PlannerChecker
from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle


class PlannerCheckerTest(unittest.TestCase):
    def setUp(self):
        self.checker = PlannerChecker()

    def test_reset(self):
        self.checker.width = 3
        self.checker.height = 3
        value = self.checker._reset("0")
        expected = [
            ["0", "0", "0"],
            ["0", "0", "0"],
            ["0", "0", "0"],
        ]
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_valid_point(self):
        self.checker.width = 3
        self.checker.height = 3
        points = [(-1, 0), (0, 0), (0, 1), (2, 1), (2, 2), (3, 2), (20, 20), (-30, -30)]
        value = []
        for point in points:
            value.append(self.checker._check_valid_point(point[0], point[1]))
        expected = [False, True, True, True, True, False, False, False]
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_diff(self):
        obs_lst = Obstacles([])
        value = self.checker.get_map_difficulity(
            obstacles=obs_lst, height=7, width=7, sx=0, sy=0, gx=6, gy=6
        )

        expected = 6
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_map_construction(self):
        obstacles = Obstacles([SingleObstacle(2, 2, 2, 1)])
        self.checker.height = 7
        self.checker.width = 7
        self.checker._construct_map(obstacles=obstacles)
        value = self.checker.map
        expected = [
            [".", ".", ".", ".", ".", ".", "."],  # 0
            [".", ".", ".", ".", ".", ".", "."],  # 1
            [".", ".", "X", "X", ".", ".", "."],  # 2
            [".", ".", ".", ".", ".", ".", "."],  # 3
            [".", ".", ".", ".", ".", ".", "."],  # 4
            [".", ".", ".", ".", ".", ".", "."],  # 5
            [".", ".", ".", ".", ".", ".", "."],  # 6
        ]
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_diff_with_obstacles(self):
        obs_lst = Obstacles([SingleObstacle(2, 2, 2, 1)])
        value = self.checker.get_map_difficulity(
            obstacles=obs_lst, height=7, width=7, sx=0, sy=0, gx=6, gy=6
        )
        expected = 7
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    # 0 1 2 3 4 5 6
    # 0 . . . . . . .
    # 1 . . . . . . .
    # 2 . . . . . . .
    # 3 . . . . . . .
    # 4 . . . . . . .
    # 5 . . . . . . .
    # 6 . . . . . . .
