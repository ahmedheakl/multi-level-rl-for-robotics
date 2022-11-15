import unittest
from highrl.utils.planner_checker import PlannerChecker, get_region_coordinates, check_valid_path_existance
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

    
    def test_region_coords(self):
        value = sorted(get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 5])[0])
        expected = sorted([[3.5, 5.5], [4.5, 4.5], [1.5, 3.5], [2.5, 2.5]])
        for p in range(len(expected)):
            self.assertListEqual(expected[p], value[p], msg=f"\nExpected:\n{expected[p]}\nFound:\n{value[p]}")
            
    def test_region_lines(self):
        _, lines = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 5])
        input_coords = [3.5, 3.5, 4.5, 2.5]
        expected = [5.5, 5.5, 4.5, 2.5]
        points = [lines[i](input_coords[i])[1] for i in range(len(lines))]
        self.assertListEqual(expected, points, msg=f"\nExpected:\n{expected}\nFound:\n{points}")
        
    def test_valid_path_existance_true(self):
        _, lines = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 5])
        obstacles = Obstacles([SingleObstacle(1, 4, 2, 1)])
        value = check_valid_path_existance(obstacles, lines, 1280, 720)
        expected = True
        self.assertEqual(expected, value, msg=f"\nExpected: {expected} || Found: {value}")
    
    def test_valid_path_existance_false(self):
        _, lines = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 5])
        obstacles = Obstacles([SingleObstacle(1, 4, 3.5, 0.5)])
        value = check_valid_path_existance(obstacles, lines, 1280, 720)
        expected = False
        self.assertEqual(expected, value, msg=f"\nExpected: {expected} || Found: {value}")
        