from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy', 'w'])  # should we add rotation ??!
ActionRot = namedtuple('ActionRot', ['v', 'r'])
