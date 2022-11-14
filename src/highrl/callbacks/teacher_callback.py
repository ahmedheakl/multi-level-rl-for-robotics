import numpy as np
import time
from pandas import DataFrame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class TeacherLogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param logpath: (string) where to save the training log
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, train_env, logpath=None, eval_freq=1, verbose=0):
        super(TeacherLogCallback, self).__init__(verbose)
        # self.model = None  # type: BaseRLModel
        self.training_env = train_env  # type: ignore
        self.logpath = logpath
        self.eval_freq = eval_freq
        self.last_len_statistics = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # get episode_statistics
            env = self.training_env
            if isinstance(self.training_env, VecEnv):
                S = DataFrame()
                for env in self.training_env.envs:  # type: ignore
                    S = S.append(env.episode_statistics, ignore_index=True)
            else:
                S = env.episode_statistics  # type: ignore

            new_S = S[self.last_len_statistics :]  # type: ignore
            new_avg_reward = np.mean(new_S["reward"].values)
            save_log(S, self.logpath, self.verbose)

            self.last_len_statistics = len(S)
        return True


def save_log(S, logpath, verbose):
    # save log
    if logpath is not None:
        S.to_csv(logpath)
        if verbose > 1:
            print("log saved to {}.".format(logpath))