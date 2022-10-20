import numpy as np
from numpy.linalg import norm
import abc

from utils.action import ActionXY


class Agent(object):
    def __init__(self, px=0, py=0, gx=0, gy=0, gt=0, vx=0, vy=0, w=0, theta=0, radius=None, config=None, section=None):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        self.radius = radius
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta

    def set(self, px, py, gx, gy, gt, vx, vy, w, theta, radius=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta
        if radius is not None:
            self.radius = radius

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

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
        while (x < 0):
            x += mod
        while (x >= mod):
            x -= mod
        return x

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        v = (action.vx**2 + action.vy**2) ** 0.5
        angle = self.fix(np.arctan2(action.vy, action.vx), 2*np.pi)
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
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius
