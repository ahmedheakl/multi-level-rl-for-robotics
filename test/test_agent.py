import unittest
from highrl.agents.agent import Agent
from highrl.utils.action import ActionXY
import numpy as np
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.agents.robot import Robot


class AgentTest(unittest.TestCase):
    def test_agent_velocity_update(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0, vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(1, 0.5, 15 * np.pi)
        delta_t = 0.2
        agent.step(action=action, delta_t=delta_t)
        value = [
            agent.vx,
            agent.vy,
            agent.px,
            agent.py,
            agent.theta,
            agent.reached_destination() == True,
        ]
        expected = [1, 0.5, 0.2, 0.1, np.pi, True]
        for value_index in range(len(value) - 1):
            self.assertAlmostEqual(
                expected[value_index],
                value[value_index],
                msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}",
            )
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}"
        )

    def test_agent_negative_x_velocity(self):
        agent = Agent(
            px=0, py=0, gx=10, gy=10, gt=0, vx=0, vy=0, w=0, theta=0, radius=10
        )
        action = ActionXY(-0.5, 1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [
            agent.vx,
            agent.vy,
            agent.px,
            agent.py,
            agent.reached_destination() == True,
        ]
        expected = [-0.5, 1, -0.1, 0.2, True]
        for value_index in range(len(value) - 1):
            self.assertAlmostEqual(
                expected[value_index],
                value[value_index],
                msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}",
            )
        self.assertEqual(
            expected[-1],
            value[-1],
            msg=f"Destination -> Expected: {expected[-1]}, Found: {value[-1]}",
        )

    def test_agent_negative_y_velocity(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0, vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(0.5, -1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [
            agent.vx,
            agent.vy,
            agent.px,
            agent.py,
            agent.reached_destination() == True,
        ]
        expected = [0.5, -1, 0.1, -0.2, True]
        for value_index in range(len(value) - 1):
            self.assertAlmostEqual(
                expected[value_index],
                value[value_index],
                msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}",
            )
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}"
        )

    def test_agent_negative_x_y_velocity(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0, vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(-0.5, -1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [
            agent.vx,
            agent.vy,
            agent.px,
            agent.py,
            agent.reached_destination() == True,
        ]
        expected = [-0.5, -1, -0.1, -0.2, True]
        for value_index in range(len(value) - 1):
            self.assertAlmostEqual(
                expected[value_index],
                value[value_index],
                msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}",
            )
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}"
        )

    def test_reached_destination_true(self):
        agent = Agent(gx=20, gy=3)
        value = agent.reached_destination()
        expected = True
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_reached_destination_false(self):
        agent = Robot()
        agent.set(gx=20, gy=25)
        value = agent.reached_destination()
        expected = False
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_collision_detection_true(self):
        agent = Robot()
        obstacle = SingleObstacle(px=19, py=0, width=10, height=10)
        value = agent.is_overlapped(obstacle=obstacle)
        expected = True
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")

    def test_collision_detection_false(self):
        agent = Robot()
        obstacle = SingleObstacle(px=25, py=0, width=10, height=10)
        value = agent.is_overlapped(obstacle=obstacle)
        expected = False
        self.assertEqual(expected, value, msg=f"Expected: {expected}, Found: {value}")
