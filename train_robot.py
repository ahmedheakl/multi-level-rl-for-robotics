from envs.robot_env import RobotEnv
from stable_baselines3.ppo.ppo import PPO
from obstacle.single_obstacle import SingleObstacle
from policy.robot_feature_extractor import Robot2DFeatureExtractor  # type: ignore
import random
from obstacle.obstacles import Obstacles

if __name__ == "__main__":
    env = RobotEnv()
    env.set_robot_position(px=100, py=100, gx=100, gy=500)
    px = random.randint(0, RobotEnv.WIDTH)
    py = random.randint(0, RobotEnv.HEIGHT)
    new_width = random.randint(50, 500)
    new_height = random.randint(50, 500)
    new_obstacle = SingleObstacle(px, py, new_width, new_height)
    env.obstacles = Obstacles([new_obstacle])
    policy_kwargs = dict(features_extractor_class=Robot2DFeatureExtractor)
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=2)
    model.learn(total_timesteps=int(1e9))
    model_save_path = "model_for_2D_lidar_testing"
    model.save(model_save_path)
