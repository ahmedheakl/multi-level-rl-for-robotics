from typing import List
import unittest
from highrl.utils.planner_checker import PlannerChecker, get_region_coordinates, check_valid_path_existance, check_if_point_inside_polygen, convex_hull_compute
from highrl.obstacle.obstacles import Obstacles
from highrl.obstacle.single_obstacle import SingleObstacle
import math

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

    
    def test_region_coords_easy(self):
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
    
    @classmethod
    def float_comparer(cls, a, b, msg=None):
        if len(a) != len(b):
            raise cls.failureException(msg)
        if not all(map(lambda args: math.isclose(*args), zip(a, b))):
            raise cls.failureException(msg)
        
    def test_region_coords_hard(self):
        value = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 7])[0]
        expected = [[3.6, 7.2], [4.4, 6.8], [2.4, 2.8], [1.6, 3.2]]
        self.addTypeEqualityFunc(list, self.float_comparer)
        for p in range(len(expected)):
            self.assertEqual(expected[p], value[p], msg=f"\nExpected:\n{expected[p]}\nFound:\n{value[p]}")
            
    def test_check_if_point_inside(self):
        coords = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 7])[0]
        points: List[List[float]] = [[3,5],[2,3],[4,7],[3,6],[4,6],[5,10],[2,1]]
        value = [check_if_point_inside_polygen(p, coords) for p in points]
        expected = [True] * 5 + [False] * 2
        self.assertListEqual(expected, value, msg=f"\nExpected:\n{expected}\nFound:\n{value}")
        
    def test_valid_path_existance_true(self):
        coords, _ = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 5])
        obstacles = Obstacles([SingleObstacle(1, 4, 2, 1)])
        value = check_valid_path_existance(obstacles, coords, 1280, 720, [2, 3], [4, 5])
        expected = True
        self.assertEqual(expected, value, msg=f"\nExpected: {expected} || Found: {value}")
    
    def test_valid_path_existance_false(self):
        coords, _ = get_region_coordinates(harmonic_number=1, eps=1, coords=[2, 3, 4, 7])
        obstacles = Obstacles([SingleObstacle(1, 4, 3, 1)])
        value = check_valid_path_existance(obstacles, coords, 1280, 720, [2, 3], [4, 7])
        expected = False
        self.assertEqual(expected, value, msg=f"\nExpected: {expected} || Found: {value}")
        
    def test_convex_polygen_compute_easy(self):
        points: List[List[float]] = [[0,0],[0,2],[2,2],[1,1]]
        value = sorted(convex_hull_compute(points))
        expected = [[0,0],[0,2],[1,1],[2,2]]
        for i in range(len(points)):
            self.assertListEqual(expected[i], value[i], msg=f"\nExpected:\n{expected}\nFound:\n{value}")
        
    def test_convex_polygen_compute_hard(self):
        points: List[List[float]] = [[2, 1], [2, 5], [3, 3], [4, 3], [4, 4], [6, 3]]
        value = convex_hull_compute(points)
        value.sort()
        expected: List[List[float]] = [[2, 1], [2, 5], [4, 4], [6, 3]]
        expected.sort()
        for i in range(len(value)):
            self.assertListEqual(expected[i], value[i], msg=f"\nExpected:\n{expected}\nFound:\n{value}")
        