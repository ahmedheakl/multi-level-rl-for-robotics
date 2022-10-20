import unittest
from utils.agent import Agent
from utils.action import ActionXY
import numpy as np


class AgentTest(unittest.TestCase):

    def test_agent_velocity_update(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0,
                      vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(1, 0.5, 15 * np.pi)
        delta_t = 0.2
        agent.step(action=action, delta_t=delta_t)
        value = [agent.vx, agent.vy, agent.px,
                 agent.py, agent.theta, agent.reached_destination() == True]
        expected = [1, 0.5, 0.2, 0.1, np.pi, True]
        for value_index in range(len(value)-1):
            self.assertAlmostEqual(expected[value_index], value[value_index],
                                   msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}")
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}")

    def test_agent_negative_x_velocity(self):
        agent = Agent(px=0, py=0, gx=10, gy=10, gt=0,
                      vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(-0.5, 1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [agent.vx, agent.vy, agent.px,
                 agent.py, agent.reached_destination() == True]
        expected = [-0.5, 1, -0.1, 0.2, False]
        for value_index in range(len(value)-1):
            self.assertAlmostEqual(expected[value_index], value[value_index],
                                   msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}")
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}")

    def test_agent_negative_y_velocity(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0,
                      vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(0.5, -1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [agent.vx, agent.vy, agent.px,
                 agent.py, agent.reached_destination() == True]
        expected = [0.5, -1, 0.1, -0.2, True]
        for value_index in range(len(value)-1):
            self.assertAlmostEqual(expected[value_index], value[value_index],
                                   msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}")
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}")

    def test_agent_negative_x_y_velocity(self):
        agent = Agent(px=0, py=0, gx=1, gy=1, gt=0,
                      vx=0, vy=0, w=0, theta=0, radius=10)
        action = ActionXY(-0.5, -1, 0.1)
        delta_t = 0.2
        agent.step(action, delta_t=delta_t)
        value = [agent.vx, agent.vy, agent.px,
                 agent.py, agent.reached_destination() == True]
        expected = [-0.5, -1, -0.1, -0.2, True]
        for value_index in range(len(value)-1):
            self.assertAlmostEqual(expected[value_index], value[value_index],
                                   msg=f"Expected: {expected[value_index]}, Found: {value[value_index]}")
        self.assertEqual(
            expected[-1], value[-1], msg=f"Expected: {expected[-1]}, Found: {value[-1]}")
