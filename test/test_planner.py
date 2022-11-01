import unittest
from teacher_env import TeacherEnv
from utils.robot import Robot
from obstacle.single_obstacle import SingleObstacle


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TeacherEnv()

    def test_overlapp(self):
        robot = Robot()
        robot.set(20, 20, 0, 0, 0, 0, 0, 0, 0, 10)
        obstacle = SingleObstacle(25, 5, 10, 30)
        overlap = self.planner._check_overlap(obstacle=obstacle, robot=robot)
        expected = True
        self.assertAlmostEqual(
            expected, overlap, msg=f"Expected: {expected}, Found: {overlap}"
        )
