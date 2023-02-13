"""
Callbacks for robot training.

Contains the following classes:
    RobotMaxStepsCallback Class: Terminates robot training after a max number of steps
    RobotLogCallback Class: Collects robot training statistics
    RobotEvalCallback Class: Evaluates robot model every certain number of steps
"""
from typing import Optional
import time
import numpy as np
from torch import nn
import gym
from pandas import DataFrame, concat
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


THINK_EMOJI = "\U0001F914"


class RobotMaxStepsCallback(BaseCallback):
    """Custom callback that derives from ``BaseCallback``"""

    def __init__(self, max_steps: int = int(1e6), verbose: int = 0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        """This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        total_steps = self.training_env.get_attr(attr_name="total_steps")[0]  # type: ignore

        if total_steps >= self.max_steps:
            print(
                f"{THINK_EMOJI} {THINK_EMOJI} Abort_training {THINK_EMOJI} {THINK_EMOJI}"
            )
            return False
        return True


class RobotSuccessesCallback(BaseCallback):
    """Callback for checking for the number of successes"""

    def __init__(self, num_successes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.num_successes = num_successes

    def _on_step(self) -> bool:
        """Check if robot reached desired number of successes

        Returns:
            bool: Flag for aborting training early
        """
        attribute_name = "num_successes"
        total_num_successes = self.training_env.get_attr(attribute_name)[0]  # type: ignore
        if total_num_successes >= self.num_successes:
            print(
                f"{THINK_EMOJI} {THINK_EMOJI} Abort_training {THINK_EMOJI} {THINK_EMOJI}"
            )
            return False
        return True


class RobotLogCallback(BaseCallback):
    """Prints and Saves training logs for the robot.

    Args:
        BaseCallback (Class): Base class for callback in stable_baselines3.
    """

    def __init__(
        self,
        train_env,
        logpath: Optional[str] = None,
        eval_frequency: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.training_env = train_env
        self.logpath = logpath
        self.eval_frequency = eval_frequency
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.last_eval_time = time.time()
        self.total_time:float = 0

    def _on_step(self) -> bool:
        """print statistics and save training logs every eval_frequency timesteps.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """

        if self.eval_frequency > 0 and self.n_calls % self.eval_frequency == 0:
            env = self.training_env
            if isinstance(self.training_env, VecEnv):
                train_logs = DataFrame()
                for env in self.training_env.envs:
                    train_logs = concat(
                        [train_logs, DataFrame.from_records(env.episode_statistics)]
                    )
            else:
                train_logs = env.episode_statistics

            last_added_train_logs = train_logs[self.last_len_statistics :]
            elapsed = time.time() - self.last_eval_time
            self.total_time += elapsed
            save_logs(train_logs, self.logpath, self.verbose)
            if self.verbose:
                print_statistics_for_n_steps(
                    last_added_train_logs,
                    elapsed_time=elapsed,
                    num_steps=10000,
                )
            if self.n_calls % (self.eval_frequency * 10) == 0 and self.verbose:
                print_statistics_train(
                    train_logs,
                    self.total_time,
                    self.n_calls,
                    self.num_timesteps,
                )

            self.last_len_statistics = len(train_logs)
            self.last_eval_time = time.time()
        return True


class RobotEvalCallback(BaseCallback):
    """Runs evaluation episodes on the trained model,save the evaluation logs
    and saves the model if improved in evaluation.

    Args:
        BaseCallback (Class): Base class for callback in stable_baselines3.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes=100,
        logpath: Optional[str] = None,
        savepath: Optional[str] = None,
        eval_frequency: int = 50000,
        verbose: int = 1,
        render=False,
    ) -> None:
        super().__init__(verbose)

        self.logpath = logpath
        self.n_eval_episodes = n_eval_episodes
        self.eval_frequency = eval_frequency
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.eval_env = eval_env
        self.savepath = savepath
        self.last_eval_time = time.time()
        self.render = render

    def _on_step(self) -> bool:
        """Runs evaluation episodes on the trained model every eval_frequency timesteps,
        saves the evaluation logs and saves the model if improved in evaluation.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """

        if self.eval_frequency > 0 and self.n_calls % self.eval_frequency == 0:
            tic = time.time()
            eval_logs = run_n_episodes(self.model, self.eval_env, self.n_eval_episodes)
            toc = time.time()
            eval_duration = toc - tic
            last_added_eval_logs = eval_logs[self.last_len_statistics :]
            new_avg_reward = np.mean(last_added_eval_logs["reward"].values)

            save_logs(eval_logs, self.logpath, self.verbose)
            if self.verbose:
                print_statistics_eval(last_added_eval_logs, eval_duration)
            self.save_model_if_improved(new_avg_reward, self.model, self.savepath)

            self.last_len_statistics = len(eval_logs)
        return True

    def save_model_if_improved(
        self,
        new_avg_reward: float,
        model: nn.Module,
        savepath: Optional[str],
    ) -> None:
        """Save the model if the average reward improved in evaluation.

        Args:
            new_avg_reward (float): New average reward in evaluation
            model (_type_): Trained model instance
            savepath (str): Path to save the model
        """
        if new_avg_reward <= self.best_avg_reward[0]:
            return

        self.best_avg_reward[0] = new_avg_reward
        if savepath is not None:
            try:
                model.save(savepath)
                print(f"Model saved to {savepath} (avg reward: {new_avg_reward}).")
            except AttributeError:
                print("Could not save")
            else:
                print("An error occured while saving the model")


def save_logs(training_logs: DataFrame, logpath: Optional[str], verbose: int) -> None:
    """Saves the training logs of the robot.

    Args:
        training_logs (DataFrame): Contains robot training logs.
        logpath (str): Path to save the logs to it.
        verbose (int): Flag to print the saving path.
    """
    training_logs.to_csv(logpath)
    if verbose > 1:
        print(f"Logs saved to {logpath}")


def print_statistics_train(
    train_logs: DataFrame,
    elapsed_time: float,
    n_calls: int,
    n_train_steps: int,
) -> None:
    """Print statistics of training.


    Args:
        train_logs (DataFrame): Contains robot training logs.
        elapsed (float): Time elapsed in training.
        n_calls (int): Number of training steps per env.
        n_train_steps (int): Total number of training steps.
    """
    scenarios = sorted(list(set(train_logs["scenario"].values)))
    rewards = train_logs["reward"].values
    print(f"Statistics for {n_train_steps:10d} steps")
    num_episodes = len(train_logs)
    stats = f"Steps Per Env = {n_calls:5d} ,"
    stats += f"Total Steps = {n_train_steps:5d} ,"
    stats += f"Total episodes = {num_episodes:5d}"
    stats += f"done in {elapsed_time} sec"
    print(stats)
    for scenario in scenarios:
        is_scenario = train_logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        avg_scenario_rewards = np.mean(scenario_rewards).item()
        num_scenarios = len(scenario_rewards)
        print(f"Avg rewards (train): {avg_scenario_rewards:.4f} ({num_scenarios}")


def print_statistics_for_n_steps(
    logs: DataFrame,
    elapsed_time: float,
    num_steps: int = 10000,
) -> None:
    """Print training stats(`num_eposides`, `avg_rewards`)
    for n steps.

    Args:
        logs (DataFrame): Contains training logs for the last n timesteps.
        elapsed_time (float): Time elapsed in last n training timesteps.
        n (int, optional): Number of training steps to print. Defaults to 10000.
    """
    scenarios = sorted(list(set(logs["scenario"].values)))
    rewards = logs["reward"].values
    # print statistics
    print(f"Statistics for last {num_steps} steps")
    num_eposides = len(logs)
    print(f"Total episodes = {num_eposides:5d} done in {elapsed_time} secs")
    for scenario in scenarios:
        is_scenario = logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        avg_scenario_rewards = np.mean(scenario_rewards).item()
        num_scenarios = len(scenario_rewards)
        print(f"Avg reward (train): {avg_scenario_rewards:.4f} ({num_scenarios})")


def run_n_episodes(
    model: nn.Module,
    env: gym.Env,
    num_eposides: int,
) -> DataFrame:
    """Run n evaluation-episodes on the trained model.

    Args:
        model (torch.nn.Module): Trained model.
        env (gym.Env): Evaluation env.
        num_eposides (int): Number of episodes to evaluate on.

    Returns:
        DataFrame: Logs for evaluation episodes.
    """
    env.episode_statistics["scenario"] = "robot_env_test"
    for _ in range(num_eposides):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
    return env.episode_statistics


def print_statistics_eval(
    eval_logs: DataFrame,
    eval_elapsed: float,
) -> None:
    """Print statistics of evaluation

    Args:
        eval_logs (DataFrame): ontains robot evaluation logs.
        eval_elapsed (float):  time elapsed in evaluation.
    """
    scenarios = sorted(list(set(eval_logs["scenario"].values)))
    rewards = eval_logs["reward"].values
    print(f"Evaluation time : {eval_elapsed} ")
    for scenario in scenarios:
        is_scenario = eval_logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        avg_scenario_rewards = np.mean(scenario_rewards).item()
        num_scenarios = len(scenario_rewards)
        print(f"{scenario}: {avg_scenario_rewards:.4f} ({num_scenarios})")
