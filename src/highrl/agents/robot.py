"""Implementation for Robot code"""
from highrl.agents.agent import Agent


class Robot(Agent):
    """
    Class that represents the robot interacting in the environment.

    This class inherits from the Agent Class.
    """

    def __init__(self, *param):
        super().__init__(*param)
