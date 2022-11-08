from highrl.agents.agent import Agent


class Robot(Agent):
    def __init__(self, *param, config=None, section=None):
        super().__init__(*param)

    # def act(self, ob):
    #     if self.policy is None:
    #         raise AttributeError('Policy attribute has to be set!')
    #     state = JointState(self.get_full_state(), ob)
    #     action = self.policy.predict(state)
    #     return action
