import numpy as np
import time
from pandas import DataFrame
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


class RobotLogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param logpath: (string) where to save the training log
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, train_env, logpath=None, eval_freq=10000, verbose=0):
        super(RobotLogCallback, self).__init__(verbose)
        # self.model = None  # type: BaseRLModel
        self.training_env = train_env  # type: ignore
        self.logpath = logpath
        self.eval_freq = eval_freq
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.last_eval_time = time.time()
        self.total_time = 0

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

            new_S = S[self.last_len_statistics :]
            new_avg_reward = np.mean(new_S["reward"].values)
            elapsed = time.time() - self.last_eval_time
            self.total_time += elapsed
            save_log(S, self.logpath, self.verbose)
            print_statistics_10K(new_S, elapsed, self.verbose)
            if self.n_calls % (self.eval_freq * 10) == 0:
                print_statistics_train(
                    S, self.total_time, self.n_calls, self.num_timesteps, self.verbose
                )

            self.last_len_statistics = len(S)
            self.last_eval_time = time.time()
        return True


def save_log(S, logpath, verbose):
    # save log
    if logpath is not None:
        S.to_csv(logpath)
        if verbose > 1:
            print("log saved to {}.".format(logpath))


def print_statistics_train(S, elapsed, n_calls, n_train_steps, verbose):
    scenarios = sorted(list(set(S["scenario"].values)))
    rewards = S["reward"].values
    # print statistics
    if verbose > 0:
        print("Statistics for {:10d} steps".format(n_train_steps))
        print(
            "Steps Per Env = {:5d} ,Total Steps = {:5d} ,Total episodes = {:5d} done in {} sec  ".format(
                n_calls, n_train_steps, len(S), elapsed
            )
        )
        for scenario in scenarios:
            is_scenario = S["scenario"].values == scenario
            scenario_rewards = rewards[is_scenario]
            print(
                "Reward in training : {:.4f} ({})".format(
                    np.mean(scenario_rewards), len(scenario_rewards)
                )
            )


def print_statistics_10K(S, elapsed, verbose):
    scenarios = sorted(list(set(S["scenario"].values)))
    rewards = S["reward"].values
    # print statistics
    if verbose > 0:
        print("Statistics for last 10K steps")
        print("Total episodes = {:5d} done in {} sec ".format(len(S), elapsed))
        for scenario in scenarios:
            is_scenario = S["scenario"].values == scenario
            scenario_rewards = rewards[is_scenario]
            print(
                "Reward in training : {:.4f} ({})".format(
                    np.mean(scenario_rewards), len(scenario_rewards)
                )
            )


def run_n_episodes(model, env, n):
    env.episode_statistics["scenario"] = "robot_env_test"
    for i in range(n):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
    return env.episode_statistics


class RobotEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param eval_env: (gym.Env) environment with which to evaluate the model (at eval_freq)
    :param test_env_fn: (function) function which returns an environment which is used to evaluate
                        the model after every tenth evaluation.
    :param n_eval_episodes: (int) how many episodes to run the evaluation env for
    :param logpath: (string) where to save the training log
    :param savepath: (string) where to save the model
    :param eval_freq: (int) how often to run the evaluation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param render: (bool) human rendering in the test env
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes=100,
        logpath=None,
        savepath=None,
        eval_freq=50000,
        verbose=1,
        render=False,
    ):
        super(RobotEvalCallback, self).__init__(verbose)

        self.logpath = logpath
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.eval_env = eval_env
        self.savepath = savepath
        self.last_eval_time = time.time()
        self.render = render

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # get episode_statistics
            tic = time.time()
            S = run_n_episodes(self.model, self.eval_env, self.n_eval_episodes)
            toc = time.time()
            eval_duration = toc - tic
            new_S = S[self.last_len_statistics :]
            new_avg_reward = np.mean(new_S["reward"].values)

            save_log(S, self.logpath, self.verbose)
            print_statistics(new_S, eval_duration, self.verbose)
            self.save_model_if_improved(new_avg_reward, self.model, self.savepath)

            self.last_len_statistics = len(S)
        return True

    def save_model_if_improved(self, new_avg_reward, model, savepath):
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
                except:  # noqa
                    print("Could not save")


def print_statistics(S, eval_elapsed, verbose):
    scenarios = sorted(list(set(S["scenario"].values)))
    rewards = S["reward"].values
    # print statistics
    if verbose > 0:
        print("Evalution time : {} ".format(eval_elapsed))
        for scenario in scenarios:
            is_scenario = S["scenario"].values == scenario
            scenario_rewards = rewards[is_scenario]
            print(
                "{}: {:.4f} ({})".format(
                    scenario, np.mean(scenario_rewards), len(scenario_rewards)
                )
            )
