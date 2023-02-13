"""
Callbacks for teacher training.

Contains the following classes:
    TeacherMaxStepsCallback Class: Terminates teacher training after a max number of steps
    TeacherLogCallback Class: Collects teacher training statistics
    TeacherSaveModelCallback Class: Saves teacher model every certain number of steps
"""
from typing import Optional
import os
import gym
from pandas import DataFrame, concat
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class TeacherMaxStepsCallback(BaseCallback):
    """
    Terminates teacher training after a max number of steps.

    This is a custom callback that derives from ``BaseCallback``.

    Args:
        BaseCallback: BaseCallback provided by stablebaselines3 to create custom callbacks

    Attributes:
        max_steps (int): max steps at which the callback terminates training
    """

    def __init__(self, max_steps: int = int(1e6), verbose: int = 0) -> None:
        """
        Constructs a TeacherMaxStepsCallback object.

        Args:
            max_steps (int, optional): _description_. Defaults to 1e6.
            verbose (int, optional): Verbosity level: 0 for no output, 1 for info messages,
                2 for debug messages. Defaults to 0.
        """
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        """
        Stops training if teacher total steps exceeded the max allowed steps.

        This method will be called by the model after each call to `env.step()`.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        total_steps = self.training_env.get_attr(attr_name="time_steps")[0]  # type: ignore
        if total_steps >= self.max_steps:
            return False
        return True


class TeacherLogCallback(BaseCallback):
    """
    Saves collected teacher training statistics in a csv file.

    This is a custom callback that derives from ``BaseCallback``.

    Args:
        BaseCallback: BaseCallback provided by stablebaselines3 to create custom callbacks

    Attributes:
        train_env (TeacherEnv): teacher train environment
        logpath (str): where to save the training log. Defaults to None.
        save_freq (int): frequency of steps at which the log file is updated. Defaults to 1.
        last_len_statistics (int): number of rows of collected data in log file
    """

    def __init__(
        self,
        train_env: gym.Env,
        logpath: Optional[str] = None,
        save_freq: int = 1,
        verbose: int = 0,
    ) -> None:
        """
        Constructs a TeacherLogCallback object.

        Args:
            train_env (TeacherEnv): teacher train environment
            logpath (str, optional): where to save the training log. Defaults to None.
            save_freq (int, optional): frequency of steps at which the log file is updated.
            Defaults to 1.
            verbose (int, optional): Verbosity level: 0 for no output, 1 for info messages,
                2 for debug messages. Defaults to 0.
        """
        super().__init__(verbose)
        # self.model = None  # type: BaseRLModel
        self.training_env = train_env
        self.logpath = logpath
        self.eval_freq = save_freq
        self.last_len_statistics = 0

    def _on_step(self) -> bool:
        """
        Saves collected statistics of teacher training every certain number of steps.

        This method will be called by the model after each call to `env.step()`. It checks to
        see if there are any additional statistics collected by the environment and if so, updates
        the log csv file.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # get session_statistics
            env = self.training_env
            if isinstance(self.training_env, VecEnv):
                session = DataFrame()

                for env in self.training_env.envs:  # type: ignore
                    if len(env.session_statistics) != 0:
                        session = concat(
                            [session, DataFrame.from_records(env.session_statistics)]
                        )

                        # new_session = session[self.last_len_statistics:]
                        # new_avg_reward = np.mean(new_S["teacher_reward"].values)
                        if self.logpath:
                            save_log(session, self.verbose, self.logpath)
            else:
                session = env.session_statistics
                # new_session = session[self.last_len_statistics :]
                # new_avg_reward = np.mean(new_S["teacher_reward"].values)
                assert (
                    self.logpath
                ), "You have to provide a path to save the session logs"
                save_log(session, self.logpath, self.verbose)

            self.last_len_statistics = len(session)
        return True


class TeacherSaveModelCallback(BaseCallback):
    """Implementation for base callback to save model weights"""

    def __init__(
        self,
        train_env: gym.Env,
        save_path: Optional[str] = None,
        save_freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        # self.model = None  # type: BaseRLModel
        self.training_env = train_env  # type: ignore
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecEnv):
                for env in self.training_env.envs:  # type: ignore
                    model_save_path = os.path.join(self.save_path, f"{env.time_steps}")
                    self.model.save(path=model_save_path)
        return True


def save_log(
    session,
    verbose: int,
    logpath: Optional[str] = None,
):
    """Save statistics for played session

    Args:
        session (_type_): Session to save statistics for
        verbose (int): Verbose level
        logpath (Optional[str], optional): Path to save logs. Defaults to None.
    """
    if logpath is None:
        return
    session.to_csv(logpath)
    if verbose > 1:
        print(f"log saved to {logpath}.")
