"""Implementation of action type for robot"""
from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy', 'w']) 
ActionRot = namedtuple('ActionRot', ['v', 'r'])
