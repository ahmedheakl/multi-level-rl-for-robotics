from stable_baselines3.common.callbacks import BaseCallback


class RobotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, max_steps=1e6, verbose=0):
        super(RobotCallback, self).__init__(verbose)
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


# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback

# def callback(save_freq, save_path, eval_env, env, reward_threshold, eval_freq):

#     checkpoint_callback = CheckpointCallback(save_freq, save_path)
#     # Separate evaluation env
#     eval_env = eval_env

#     env = env

#     # Stop training when the model reaches the reward threshold
#     callback_on_best = StopTrainingOnRewardThreshold(reward_threshold, verbose=1)
#     eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1, eval_freq = eval_freq,
#                                  render = True)

#     callback = CallbackList([checkpoint_callback, eval_callback])

#     return callback
