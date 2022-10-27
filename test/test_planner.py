import unittest
from planner_env import PlannerEnv
from utils.robot import Robot
from obstacle.singleobstacle import SingleObstacle

class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = PlannerEnv()

    def test_overlapp(self):
        robot = Robot()
        robot.set(20.0, 20.0, 0, 0, 0, 0, 0, 0, 0, 10.0)
        obstacle = SingleObstacle(25, 5, 10, 30)
        nooverlap = self.planner._check_overlap(obstacle=obstacle, robot=robot)
        expected = False
        self.assertAlmostEqual(expected, nooverlap, msg=f"Expected: {expected}, Found: {nooverlap}")