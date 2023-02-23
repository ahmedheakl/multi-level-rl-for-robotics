"""Implementation for agents interface"""
from typing import Any, Tuple, List, Union
import numpy as np
from numpy.linalg import norm

from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.utils.action import ActionXY
from highrl.utils.abstract import Position


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
        pos=Position[float](0.0, 0.0),
        goal_pos=Position[float](0.0, 0.0),
        gt: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        w: float = 0.0,
        theta: float = 0.0,
        radius: int = 20,
        goal_radius: int = 10,
    ) -> None:
        """Constructs an agent object.

        Args:
            pos (Position, optional): Position of agent. Defaults to (x=0, y=0).
            gpos (Position, optional): Position of the goal. Defaults to (x=0, y=0).
            gt (int, optional): Goal orientation angle. Defaults to 0.
            vx (int, optional): Agent x velocity. Defaults to 0.
            vy (int, optional): Agent y velocity. Defaults to 0.
            w (int, optional): Agent angular velocity. Defaults to 0.
            theta (int, optional): Agent angle theta. Defaults to 0.
            radius (int, optional): Agent radius. Defaults to 20.
            goal_radius (int, optional): Goal radius. Defaults to 10.
        """
        self.radius = radius
        self.goal_radius = goal_radius
        self.pos = pos
        self.gpos = goal_pos
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta

    def set(
        self,
        pos=Position[float](0.0, 0.0),
        goal_pos=Position[float](0.0, 0.0),
        gt: float = 0,
        vx: float = 0,
        vy: float = 0,
        w: float = 0,
        theta: float = 0,
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
        self.pos = pos
        self.gpos = goal_pos
        self.gt = gt
        self.vx = vx
        self.vy = vy
        self.w = w
        self.theta = theta
        self.radius = radius
        self.goal_radius = goal_radius

    @property
    def x_pos(self) -> float:
        """Getter for x_coord"""
        return self.pos.x

    @property
    def y_pos(self) -> float:
        """Getter for y_coord"""
        return self.pos.y

    def get_position(self) -> Position:
        """Gets the agent postion"""
        return self.pos

    def set_position(self, position: Position) -> None:
        """Sets agent position"""
        self.pos = position

    def set_goal_position(self, position: Position) -> None:
        """Sets goal position"""
        self.gpos = position

    def get_goal_position(self) -> Position:
        """Gets the goal postion"""
        return self.gpos

    def get_velocity(self) -> Tuple[float, float, float]:
        """Gets agent velocity vector.

        Returns:
            Tuple[float, float, float]: (agent x velocity, agent y velocity, agent angular velocity)
        """
        return self.vx, self.vy, self.w

    def set_velocity(self, velocity: Tuple[float, ...]):
        """Sets agent linear and angular velocity.

        Args:
            velocity (Tuple[int, int]): (agent x velocity, agent y velocity, agent angular velocity)
        """
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.w = velocity[2]

    def set_radius(self, agent_radius: int, goal_radius: int) -> None:
        """Setter for the goal and agent radius"""
        self.radius = agent_radius
        self.goal_radius = goal_radius

    def check_validity(self, action: Any):
        """Checks if action is in right format.

        The right format is the object forman: ActionXY
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
        velocity = (action.vx**2 + action.vy**2) ** 0.5
        angle = self.fix(np.arctan2(action.vy, action.vx), 2 * np.pi)
        x_pos = self.pos.x + velocity * np.cos(self.theta + angle) * delta_t
        y_pos = self.pos.y + velocity * np.sin(self.theta + angle) * delta_t
        theta = self.fix(self.theta + action.w * delta_t, 2 * np.pi)

        return x_pos, y_pos, theta

    def fix(self, base: Union[int, float], mod: Union[int, float]) -> Union[int, float]:
        """Fix input x to be in range [0:mod-1].

        Args:
            base (Union[int, float]): input range
            mod (int): modulus

        Returns:
            Union[int, float]: Input with desired range
        """
        while base < 0:
            base += mod
        while base >= mod:
            base -= mod
        return base

    def reached_destination(self) -> bool:
        """Determines if agent reached the goal postion.

        Returns:
            bool: whether the agent has reached the goal or not
        """
        min_allowed_dist = self.radius + self.goal_radius
        return self.dist_to_goal() < min_allowed_dist

    def dist_to_goal(self) -> float:
        """Compute the distance from the agent to the goal"""
        return norm(self.pos.get_coords() - self.gpos.get_coords()).item()

    def step(self, action: ActionXY, delta_t: float) -> None:
        """Performs an action and update the agent state.
        Args:
            action (List): action decided by the agent model but in ActionXY object format
            delta_t (float): time difference between actions
        """
        self.check_validity(action)
        pos = self.compute_position(action, delta_t)
        x_pos, y_pos, self.theta = pos
        self.pos.set_pos(x_pos, y_pos)
        self.vx = action.vx
        self.vy = action.vy
        self.w = action.w

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
        ], "Check target should be goal or agent"
        if check_target == "goal":
            min_x = self.pos.x - self.goal_radius
            min_y = self.pos.x - self.goal_radius
            max_x = self.gpos.x + self.goal_radius
            max_y = self.gpos.y + self.goal_radius
        else:
            min_x = self.pos.x - self.radius
            min_y = self.pos.y - self.radius
            max_x = self.pos.x + self.radius
            max_y = self.pos.y + self.radius

        dummy = [
            [
                min_x,
                min_y,
                max_x,
                max_y,
            ],
            [
                int(obstacle.px),
                int(obstacle.py),
                int(obstacle.px + obstacle.width),
                int(obstacle.py + obstacle.height),
            ],
        ]
        is_overlap: bool = not self._overlap_handler(dummy)
        return is_overlap

    def is_robot_overlap_goal(self) -> bool:
        """Check if robot and goal overlap.

        Returns:
            bool: flag to check for overlap. Returns True if there is an overlap.
        """
        dummy = [
            [
                self.gpos.x - self.goal_radius,
                self.gpos.x - self.goal_radius,
                self.gpos.x + self.goal_radius,
                self.gpos.x + self.goal_radius,
            ],
            [
                self.pos.x - self.radius,
                self.pos.y - self.radius,
                self.pos.x + self.radius,
                self.pos.y + self.radius,
            ],
        ]
        is_overlap = not self._overlap_handler(dummy)
        return is_overlap

    def is_robot_close_to_goal(self, min_dist: int) -> bool:
        """Checks if the robot is closer than the min distannce to the goal.

           Returns ``True`` if the robot is too close and ``False`` if the robot-goal dist
           did not exceed the min allowed distance.

        Args:
            min_dist (int): min allowable distance for robot-goal dist

        Returns:
            bool: flag to determine if the robot is closer than the max allowed distance or not.
        """
        distance = self.dist_to_goal()
        distance -= self.radius + self.goal_radius
        return distance <= min_dist

    def _overlap_handler(self, dummy: List[List]) -> bool:
        """Check overlap condition between two objects.

        Args:
            dummy (List[List]): objects coordinates

        Returns:
            bool: overlap flag for input objects
        """
        for _ in range(2):
            if dummy[0][0] > dummy[1][2] or dummy[0][1] > dummy[1][3]:
                return True
            dummy[0], dummy[1] = dummy[1], dummy[0]
        return False
