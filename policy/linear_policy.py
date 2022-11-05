import numpy as np
from policy.policy import Policy
from utils.action import ActionPos


class Linear(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.kinematics = 'holonomic'

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py,
                           self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionPos(vx, vy)

        return action
