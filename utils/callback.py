from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback

def callback(save_freq, save_path, eval_env, env, reward_threshold, eval_freq):
    
    checkpoint_callback = CheckpointCallback(save_freq, save_path)
    # Separate evaluation env
    eval_env = eval_env

    env = env

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1, eval_freq = eval_freq,
                                 render = True)

    callback = CallbackList([checkpoint_callback, eval_callback])
    
    return callback