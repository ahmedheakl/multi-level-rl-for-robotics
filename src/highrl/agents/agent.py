"""
Creates and manages agents.

Contains:
    Agent Class: generic class for humans and robots
"""
from typing import Any
import numpy as np
from numpy.linalg import norm
import abc
from typing import Tuple, List
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.utils.action import ActionXY
from highrl.utils.calculations import point_to_point_distance


class Agent:
    """
    Class that represents the agent interacting in the environment.

    Attributes:
        px (int): agent x position.
        gx (int): goal x position.
        gy (int): goal y position.
        gt (int): goal orientation angle.
        vx (int): agent x velocity.
        vy (int): agent y velocity.
        py (int): agent y position.
        w (int): agent angular velocity.
        theta (int): agent angle theta.
        radius (int): agent radius.
        goal_radius (int): goal radius.
    """

    def __init__(
        self,
        px: int = 0,
        py: int = 0,
        gx: int = 0,
        gy: int = 0,
        gt: int = 0,
        vx: int = 0,
        vy: int = 0,
        w: int = 0,
        theta: int = 0,
        radius: int = 20,
        goal_radius: int = 10,
    ) -> None:
        """Constructs an agent object.

        Args:
            px (int, optional): agent x position. Defaults to 0.
            py (int, optional): agent y position. Defaults to 0.
            gx (int, optional): goal x position. Defaults to 0.
            gy (int, optional): goal y position. Defaults to 0.
            gt (int, optional): goal orientation angle. Defaults to 0.
            vx (int, optional): agent x velocity. Defaults to 0.
            vy (int, optional): agent y velocity. Defaults to 0.
            w (int, optional): agent angular velocity. Defaults to 0.
            theta (int, optional): agent angle theta. Defaults to 0.
            radius (int, optional): agent radius. Defaults to 20.
            goal_radius (int, optional): goal radius. Defaults to 10.
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
        px: int = 0,
        py: int = 0,
        gx: int = 0,
        gy: int = 0,
        gt: int = 0,
        vx: int = 0,
        vy: int = 0,
        w: int = 0,
        theta: int = 0,
        radius: int = 20,
        goal_radius: int = 10,
    ) -> None:
        """Sets all agent attributes.

        Args:
            px (int, optional): agent x position. Defaults to 0.
            py (int, optional): agent y position. Defaults to 0.
            gx (int, optional): goal x position. Defaults to 0.
            gy (int, optional): goal y position. Defaults to 0.
            gt (int, optional): goal orientation angle. Defaults to 0.
            vx (int, optional): agent x velocity. Defaults to 0.
            vy (int, optional): agent y velocity. Defaults to 0.
            w (int, optional): agent angular velocity. Defaults to 0.
            theta (int, optional): agent angle theta. Defaults to 0.
            radius (int, optional): agent radius. Defaults to 20.
            goal_radius (int, optional): goal radius. Defaults to 10.
        """
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

    def get_position(self) -> Tuple[int, int]:
        """Gets the agent postion.

        Returns:
            Tuple(int, int): (agent x position, agent y position)
        """
        return self.px, self.py

    def set_position(self, position: Tuple[int, int, int]) -> None:
        """Sets agent position.

        Args:
            position (Tuple[int, int]): (agent x position, agent y position)
        """
        self.px, self.py = position

    def set_goal_position(self, position: Tuple[int, int]) -> None:
        """Sets goal position.

        Args:
            position (Tuple[int, int]): (goal x position, goal y position)
        """
        self.gx, self.gy = position

    def get_goal_position(self) -> Tuple[int, int]:
        """Gets the goal postion.

        Returns:
            Tuple[int, int]: (goal x position, goal y position)
        """
        return self.gx, self.gy

    def get_velocity(self) -> Tuple[int, int, int]:
        """Gets agent velocity vector.

        Returns:
            Tuple[int, int, int]: (agent x velocity, agent y velocity, agent angular velocity)
        """
        return self.vx, self.vy, self.w

    def set_velocity(self, velocity: Tuple):
        """Sets agent linear and angular velocity.

        Args:
            velocity (Tuple[int, int]): (agent x velocity, agent y velocity, agent angular velocity)
        """
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.w = velocity[2]

    def check_validity(self, action: Any):
        """Checks if action is in right format.

        The right format is the object forman: ActionXY

        Args:
            action (Any): action whose format is required to be checked

        Raises:
            AssertionError: raises error if the action is not in the ActionXY object format
        """
        assert isinstance(action, ActionXY)

    def compute_position(self, action: Any, delta_t: float) -> Tuple:
        """Computes agent next position and orientation based on the agent action velocity.

           Before computing the agent next position, Checks if the action is in the ActionXY
           format.

        Args:
            action (Any): action decided by the agent model but in ActionXY object format
            delta_t (float): time difference between actions

        Returns:
            Tuple[int, int, int]: (agent x position, agent y posistion, agent orientation theta)
        """
        self.check_validity(action)
        v = (action.vx**2 + action.vy**2) ** 0.5
        angle = self.fix(np.arctan2(action.vy, action.vx), 2 * np.pi)
        px = self.px + v * np.cos(self.theta + angle) * delta_t
        py = self.py + v * np.sin(self.theta + angle) * delta_t
        theta = self.fix(self.theta + action.w * delta_t, 2 * np.pi)

        return px, py, theta

    def step(self, action: List, delta_t: float) -> None:
        """Performs an action and update the agent state.

        Args:
            action (List): action decided by the agent model but in ActionXY object format
            delta_t (float): time difference between actions
        """
        self.check_validity(action)
        pos = self.compute_position(action, delta_t)
        self.px, self.py, self.theta = pos
        self.vx = action.vx
        self.vy = action.vy
        self.w = action.w

    def fix(self, x, mod):
        """Fix input x to be in range [0:mod-1].

        Args:
            x (int | float): input range
            mod (int | float): modulus

        Returns:
            int | float: input with desired range
        """
        while x < 0:
            x += mod
        while x >= mod:
            x -= mod
        return x

    def reached_destination(self) -> bool:
        """Determines if agent reached the goal postion.

        Returns:
            bool: whether the agent has reached the goal or not
        """
        robot_pos = np.array(self.get_position())
        goal_pos = np.array(self.get_goal_position())
        agent_goal_dist = robot_pos - goal_pos
        min_allowed_dist = self.radius + self.goal_radius
        return norm(agent_goal_dist) < min_allowed_dist

    def is_overlapped(self, obstacle: SingleObstacle, check_target: str = "agent"):
        """Checks if overlap between the agent/goal and an obstacle.

        Args:
            obstacle (SingleObstacle): input obstalce to check overlap with
            check_target (str): target to be checked, either agent or goal

        Returns:
            bool: flag to check for overlap. Returns True if there is overlap.
        """
        assert check_target in [
            "goal",
            "agent",
        ], f"check target should be goal or agent"
        if check_target == "goal":
            min_x = self.gx - self.goal_radius
            min_y = self.gy - self.goal_radius
            max_x = self.gx + self.goal_radius
            max_y = self.gy + self.goal_radius
        else:
            min_x = self.px - self.radius
            min_y = self.py - self.radius
            max_x = self.px + self.radius
            max_y = self.py + self.radius

        dummy = [
            [
                min_x,
                min_y,
                max_x,
                max_y,
            ],
            [
                obstacle.px,
                obstacle.py,
                obstacle.px + obstacle.width,
                obstacle.py + obstacle.height,
            ],
        ]
        return not (self._overlap_handler(dummy))

    def is_robot_overlap_goal(self) -> bool:
        """Check if robot and goal overlap.

        Returns:
            bool: flag to check for overlap. Returns True if there is an overlap.
        """
        dummy = [
            [
                self.gx - self.goal_radius,
                self.gy - self.goal_radius,
                self.gx + self.goal_radius,
                self.gy + self.goal_radius,
            ],
            [
                self.px - self.radius,
                self.py - self.radius,
                self.px + self.radius,
                self.py + self.radius,
            ],
        ]
        return not (self._overlap_handler(dummy))

    def is_robot_close_to_goal(self, min_dist: int) -> bool:
        """Checks to see if the robot is closer than the min distannce to the goal.

           Returns ``True`` if the robot is too close and ``False`` if the robot-goal dist
           did not exceed the min allowed distance.

        Args:
            min_dist (int): min allowable distance for robot-goal dist

        Returns:
            bool: flag to determine if the robot is closer than the max allowed distance or not.
        """
        distance = point_to_point_distance((self.px, self.py), (self.gx, self.gy)) - (
            self.radius + self.goal_radius
        )
        if distance <= min_dist:
            return True
        return False

    def _overlap_handler(self, dummy: List[List]) -> bool:
        """Check overlap condition between two objects.

        Args:
            dummy (List[List]): objects coordinates

        Returns:
            bool: overlap flag for input objects
        """
        for i in range(2):
            if dummy[0][0] > dummy[1][2] or dummy[0][1] > dummy[1][3]:
                return True
            dummy[0], dummy[1] = dummy[1], dummy[0]
        return False
