import numpy as np
import time
from pandas import DataFrame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class TeacherMaxStepsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, max_steps=1e6, verbose=0):
        super(TeacherMaxStepsCallback, self).__init__(verbose)
        self.max_steps = max_steps
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        total_steps = self.training_env.get_attr(attr_name="time_steps")[0]  # type: ignore
        if total_steps >= self.max_steps:
            return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


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
                    if len(env.episode_statistics) != 0:
                        S = S.append(env.episode_statistics, ignore_index=True)
                        new_S = S[self.last_len_statistics :]  # type: ignore
                        new_avg_reward = np.mean(new_S["reward"].values)
                        save_log(S, self.logpath, self.verbose)
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
