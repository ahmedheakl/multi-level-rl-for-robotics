"""This module contains:
    Robot Class: class specific for robots. This class inherits from the Agent class
"""
from highrl.agents.agent import Agent


class Robot(Agent):
    def __init__(self, *param):
        super().__init__(*param)
