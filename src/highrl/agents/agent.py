from typing import Any
import numpy as np
from numpy.linalg import norm
import abc

from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.utils.calculations import point_to_obstacle_distance
from highrl.utils.action import ActionXY


class Agent(object):
    def __init__(
        self,
        px=0,
        py=0,
        gx=0,
        gy=0,
        gt=0,
        vx=0,
        vy=0,
        w=0,
        theta=0,
        radius=20,
        goal_radius=10,
        config=None,
        section=None,
    ):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        self.radius = radius
        self.goal_radius = goal_radius
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta

    def set(
        self,
        px=0,
        py=0,
        gx=0,
        gy=0,
        gt=0,
        vx=0,
        vy=0,
        w=0,
        theta=0,
        radius=20,
        goal_radius=10,
    ):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta
        self.radius = radius
        self.goal_radius = goal_radius

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px, self.py = position

    def set_goal_position(self, position):
        self.gx, self.gy = position

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy, self.w

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.w = velocity[2]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy
        """
        return

    def check_validity(self, action):
        assert isinstance(action, ActionXY)

    def fix(self, x, mod):
        while x < 0:
            x += mod
        while x >= mod:
            x -= mod
        return x

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        v = (action.vx**2 + action.vy**2) ** 0.5
        angle = self.fix(np.arctan2(action.vy, action.vx), 2 * np.pi)
        px = self.px + v * np.cos(self.theta + angle) * delta_t
        py = self.py + v * np.sin(self.theta + angle) * delta_t
        theta = self.fix(self.theta + action.w * delta_t, 2 * np.pi)

        return px, py, theta

    def step(self, action, delta_t):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, delta_t)
        self.px, self.py, self.theta = pos
        self.vx = action.vx
        self.vy = action.vy
        self.w = action.w

    def reached_destination(self):

        return norm(
            np.array(self.get_position()) - np.array(self.get_goal_position())  # type: ignore
        ) < (self.radius + self.goal_radius)

    def is_overlapped(self, obstacle: SingleObstacle):
        """Check if there is no overlap between the robot and an obstacle

        Args:
            obstacle (SingleObstacle): input obstalce

        Returns:
            bool: flag to determine if there is no overlap
        """
        # min_x, min_y, max_x, max_y
        dummy1 = [
            [
                self.px - self.radius,
                self.py - self.radius,
                self.px + self.radius,
                self.py + self.radius,
            ],
            [
                obstacle.px,
                obstacle.py,
                obstacle.px + obstacle.width,
                obstacle.py + obstacle.height,
            ],
        ]
        dummy2 = [
            [
                self.gx - self.radius,
                self.gy - self.radius,
                self.gx + self.radius,
                self.gy + self.radius,
            ],
            [
                obstacle.px,
                obstacle.py,
                obstacle.px + obstacle.width,
                obstacle.py + obstacle.height,
            ],
        ]
        return not (self._overlap_handler(dummy1) and self._overlap_handler(dummy2))

    def _overlap_handler(self, dummy):
        """Check overlap condition between two objects

        Args:
            dummy (list[list]): objects coordinates

        Returns:
            boolean: overlap flag for input objects
        """
        for i in range(2):
            if dummy[0][0] > dummy[1][2] or dummy[0][1] > dummy[1][3]:
                return True
            dummy[0], dummy[1] = dummy[1], dummy[0]

        return False
