import numpy as np
from pyniel.python_tools.timetools import WalltimeRate
import time

SIDE_SPEED = 2.0
FRONT_SPEED = 2.0
BACK_SPEED = 2.0
ROT_SPEED = 2.0


class EnvPlayer(object):
    def __init__(self, env, step_by_step: bool = False, save_to_file: bool = False):
        """_summary_

        Args:
            env (_type_): _description_
            step_by_step (bool, optional): _description_. Defaults to False.
            save_to_file (bool, optional): _description_. Defaults to False.
        """
        self.env = env
        self.STEP_BY_STEP = step_by_step
        self.boost = False
        self.save_to_file = save_to_file
        self.run()

    def key_press(self, k, mod):
        from pyglet.window import key

        if k == key.ESCAPE:
            self.exit = True
        if k == key.R:
            self.restart = True
        if k in [key.RIGHT, key.D]:
            self.action[0] = +SIDE_SPEED
        if k in [key.LEFT, key.A]:
            self.action[0] = -SIDE_SPEED
        if k in [key.UP, key.W]:
            self.action[1] = +FRONT_SPEED
        if k in [key.DOWN, key.S]:
            self.action[1] = -BACK_SPEED
        if k in [key.E]:
            self.action[2] = +ROT_SPEED
        if k in [key.Q]:
            self.action[2] = -ROT_SPEED

        if k in [key.LSHIFT]:
            self.boost = True
        if k in [key.SPACE]:
            pass
        self.action_key_is_set = True

    def key_release(self, k, mod):  # reverse action of pressed
        from pyglet.window import key

        if k in [key.UP, key.W] and self.action[1] == +FRONT_SPEED:
            self.action[1] = 0
        if k in [key.DOWN, key.S] and self.action[1] == -BACK_SPEED:
            self.action[1] = 0
        if k in [key.RIGHT, key.D] and self.action[0] == +SIDE_SPEED:
            self.action[0] = 0
        if k in [key.LEFT, key.A] and self.action[0] == -SIDE_SPEED:
            self.action[0] = 0
        if k in [key.E] and self.action[2] == +ROT_SPEED:
            self.action[2] = 0
        if k in [key.Q] and self.action[2] == -ROT_SPEED:
            self.action[2] = 0

        if k in [key.LSHIFT]:
            self.boost = False
        if k in [key.SPACE]:
            pass

    def reset(self):
        # reset env
        print("Resetting")
        self.env.done = True
        self.env.reset()
        # reset player
        self.realtime_rate = WalltimeRate(1.0 / 0.01)
        self.action = np.array([0.0, 0.0, 0.0])
        self.restart = False
        self.exit = False
        self.action_key_is_set = False
        self.restart = False
        # reset viewer, connect callbacks
        self.env.render()
        self.env._get_viewer().window.on_key_press = self.key_press
        self.env._get_viewer().window.on_key_release = self.key_release

    def run(self):
        # run interactively ----------------------
        self.reset()

        print()
        print("-------------------")
        print("Running environment")
        print("press WASD to move, Q and E to rotate, shift to speed-up.")
        print("R to reset, ESC to exit.")
        print("...")

        i = 0
        while not self.exit:
            i += 1
            # synchronize (either with keypresses or walltime)
            if self.STEP_BY_STEP:
                # wait for keypress
                while True:
                    if self.boost:
                        break
                    if not self.action_key_is_set:
                        self.env.render(
                            mode=self.render_mode, save_to_file=self.save_to_file
                        )
                        time.sleep(0.01)
                    else:
                        self.action_key_is_set = False
                        break
            else:
                if not self.boost:
                    self.realtime_rate.sleep()
            # step once
            obs, rew, done, info = self.env.step(self.action)
            self.env.render(save_to_file=self.save_to_file)
            #         impglet
            #         pygage.get_buffer_manager().get_color_buffer().save("/tmp/env{:05}.png".format(i))
            if done or self.restart:
                self.reset()
        self.env.close()
