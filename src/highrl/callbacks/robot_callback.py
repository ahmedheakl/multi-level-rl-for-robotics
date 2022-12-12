"""
Callbacks for robot training.

Contains the following classes:
    RobotMaxStepsCallback Class: Terminates robot training after a max number of steps
    RobotLogCallback Class: Collects robot training statistics
    RobotEvalCallback Class: Evaluates robot model every certain number of steps
"""
import numpy as np
import time
from pandas import DataFrame, concat
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class RobotMaxStepsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, max_steps=1e6, verbose=0):
        super(RobotMaxStepsCallback, self).__init__(verbose)
        self.max_steps = max_steps

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
        thinking_emoji = "\U0001F914"
        total_steps = self.training_env.get_attr(attr_name="total_steps")[0]  # type: ignore
        if total_steps >= self.max_steps:
            print(
                f"{thinking_emoji} {thinking_emoji} Abort_training {thinking_emoji} {thinking_emoji}"
            )
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


class RobotSuccessesCallback(BaseCallback):
    def __init__(self, num_successes=5, verbose=0):
        super(RobotSuccessesCallback, self).__init__(verbose)
        self.num_successes = num_successes

    def _on_step(self) -> bool:
        thinking_emoji = "\U0001F914"
        total_num_successes = self.training_env.get_attr(attr_name="num_successes")[0]  # type: ignore
        if total_num_successes >= self.num_successes:
            print(
                f"{thinking_emoji} {thinking_emoji} Abort_training {thinking_emoji} {thinking_emoji}"
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
        logpath: str = None,
        eval_frequency: int = 10000,
        verbose: int = 0,
    ) -> None:
        super(RobotLogCallback, self).__init__(verbose)
        self.training_env = train_env
        self.logpath = logpath
        self.eval_frequency = eval_frequency
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.last_eval_time = time.time()
        self.total_time = 0

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
                    last_added_train_logs, elapsed=elapsed, n=10000
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
    """Runs evaluation episodes on the trained model,save the evaluation logs and saves the model if improved in evaluation.

    Args:
        BaseCallback (Class): Base class for callback in stable_baselines3.
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes=100,
        logpath: str = None,
        savepath: str = None,
        eval_frequency: int = 50000,
        verbose: int = 1,
        render=False,
    ) -> None:
        super(RobotEvalCallback, self).__init__(verbose)

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
        """Runs evaluation episodes on the trained model every eval_frequency timesteps, saves the evaluation logs and saves the model if improved in evaluation.

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
        self, new_avg_reward: float, model, savepath: str
    ) -> None:
        """saves the model if the average reward improved in evaluation.

        Args:
            new_avg_reward (float): the new average reward in evaluation.
            model (_type_): trained model.
            savepath (str): path to save the model to it.
        """

        if new_avg_reward > self.best_avg_reward[0]:
            self.best_avg_reward[0] = new_avg_reward
            if savepath is not None:
                try:
                    model.save(savepath)
                    print(
                        "model saved to {} (avg reward: {}).".format(
                            savepath, new_avg_reward
                        )
                    )
                except:
                    print("Could not save")


def save_logs(training_logs: DataFrame, logpath: str, verbose: int) -> None:
    """Saves the training logs of the robot.

    Args:
        training_logs (DataFrame): contains robot training logs.
        logpath (str): path to save the logs to it.
        verbose (int): flag to print the saving path.
    """
    if logpath is not None:
        training_logs.to_csv(logpath)
        if verbose > 1:
            print("log saved to {}.".format(logpath))


def print_statistics_train(
    train_logs: DataFrame, elapsed: float, n_calls: int, n_train_steps: int
) -> None:
    """print statistics of training.


    Args:
        train_logs (DataFrame): contains robot training logs.
        elapsed (float): time elapsed in training.
        n_calls (int): number of training steps per env.
        n_train_steps (int): total number of training steps.
    """
    scenarios = sorted(list(set(train_logs["scenario"].values)))
    rewards = train_logs["reward"].values
    print("Statistics for {:10d} steps".format(n_train_steps))
    print(
        "Steps Per Env = {:5d} ,Total Steps = {:5d} ,Total episodes = {:5d} done in {} sec  ".format(
            n_calls, n_train_steps, len(train_logs), elapsed
        )
    )
    for scenario in scenarios:
        is_scenario = train_logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        print(
            "Reward in training : {:.4f} ({})".format(
                np.mean(scenario_rewards), len(scenario_rewards)
            )
        )


def print_statistics_for_n_steps(
    logs: DataFrame, elapsed: float, n: int = 10000
) -> None:
    """_summary_

    Args:
        logs (DataFrame): contains robot training logs for the last n timesteps.
        elapsed (float): time elapsed in last n training timesteps.
        n (int, optional): number of training steps to print. Defaults to 10000.
    """
    scenarios = sorted(list(set(logs["scenario"].values)))
    rewards = logs["reward"].values
    # print statistics

    print("Statistics for last {} steps".format(n))
    print("Total episodes = {:5d} done in {} sec ".format(len(logs), elapsed))
    for scenario in scenarios:
        is_scenario = logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        print(
            "Reward in training : {:.4f} ({})".format(
                np.mean(scenario_rewards), len(scenario_rewards)
            )
        )


def run_n_episodes(model, env, n: int) -> DataFrame:
    """runs n evaluation episodes on the trained model.

    Args:
        model (_type_): trained model.
        env (_type_): evaluation env.
        n (int): number of episodes.

    Returns:
        DataFrame: logs for evaluation episodes.
    """
    env.episode_statistics["scenario"] = "robot_env_test"
    for i in range(n):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
    return env.episode_statistics


def print_statistics_eval(eval_logs: DataFrame, eval_elapsed: float) -> None:
    """print statistics of evaluation.

    Args:
        eval_logs (DataFrame): ontains robot evaluation logs.
        eval_elapsed (float):  time elapsed in evaluation.
    """
    scenarios = sorted(list(set(eval_logs["scenario"].values)))
    rewards = eval_logs["reward"].values
    print("Evaluation time : {} ".format(eval_elapsed))
    for scenario in scenarios:
        is_scenario = eval_logs["scenario"].values == scenario
        scenario_rewards = rewards[is_scenario]
        print(
            "{}: {:.4f} ({})".format(
                scenario, np.mean(scenario_rewards), len(scenario_rewards)
            )
        )
