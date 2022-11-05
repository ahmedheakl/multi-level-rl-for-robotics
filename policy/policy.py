import numpy as np
import abc


class Policy(object):
    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None

    def get_model(self):
        return self.model
    

    @abc.abstractmethod
    def predict(self, state):
        return

    @staticmethod
    def react_destination(state) -> bool:
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False
