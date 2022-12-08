"""
Creates and manages robot agents.

Contains:
    Robot Class: class specific for robots. This class inherits from the Agent class
"""
from highrl.agents.agent import Agent


class Robot(Agent):
    """
    Class that represents the robot interacting in the environment.

    This class inherits from the Agent Class.

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

    def __init__(self, *param):
        super().__init__(*param)
