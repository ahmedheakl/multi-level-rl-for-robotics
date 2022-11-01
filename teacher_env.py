from typing import List, Tuple, Any
from envs.env import RobotEnv
from gym import Env, spaces
from obstacle.obstacles import Obstacles
from obstacle.single_obstacle import SingleObstacle
import numpy as np
from utils.calculations import *
from policy.robot_feature_extractor import Robot1DFeatureExtractor
from stable_baselines3.ppo.ppo import PPO
from random import randint, random, uniform
from planner.planner import CustomActorCriticPolicy
from planner.planner import CustomLSTM
from utils.planner_checker import PlannerChecker
from utils.callback import RobotCallback
from utils.robot import Robot
from time import time

# TODO: rescale action to [0,1]
# TODO: scale the number_of_obstacles from [0, 1] to [0, max_allowed_obstacles]
# TODO: scale the observation space to [0,1]


class TeacherEnv(Env):
    ALPHA = 0.4
    MAX_REWARD = 3600
    TERMINAL_STATE_REWARD = 100
    MAX_TIMESTEPS = 100
    ADVANCE_PROBABILITY = 0.9
    BASE_DIFFICULTY = 590
    MAX_STEPS_FOR_ROBOT_EPISODE = 1e5
    MAX_OBSTACLES_COUNT = 10

    def __init__(self) -> None:
        super(TeacherEnv, self).__init__()
        self.action_space_names = ["robot_position", "goal_position", "obstacles_count"]
        self.action_space = spaces.Box(
            low=0.01, high=0.99, shape=(3,), dtype=np.float32
        )

        # [time_steps, robot_level, robot_reward, current_difficulity]
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(3,), dtype=np.float32
        )

        self.env = RobotEnv()
        self.episodes = 0
        self.current_difficulty = 0
        self.desired_difficulty = 1
        self.time_steps = 0
        self.robot_level = 0
        self.done = 0
        self.current_robot_reward = 0
        self.robot_success_flag = 0
        self.previous_save_path = ""

        self.terminal_state_flag = 0  # gamma

        self.checker = PlannerChecker()

    def step(self, action) -> Tuple:
        """Take a step in the environment

        Args:
            action (list): action to take

        Returns:
            tuple: observation, reward, done, info
        """
        self.current_robot_reward = self.env.total_reward
        self.robot_success_flag = self.env.success_flag
        self.env.reset()

        action = self._convert_action_to_dict_format(action)

        px, py, gx, gy = self._get_robot_position_from_action(action)

        self.env.set_robot_position(px=px, py=py, gx=gx, gy=gy)
        import math

        self.__generate_obstacles_points(
            math.ceil(action["obstacles_count"] * self.MAX_OBSTACLES_COUNT)
        )

        args_list = list(map(int, [px, py, gx, gy]))
        self.current_difficulty = self.checker.get_map_difficulity(
            self.env.obstacles, RobotEnv.WIDTH, RobotEnv.HEIGHT, *args_list
        )

        self.desired_difficulty = self.BASE_DIFFICULTY * (1.15) ** self.episodes

        policy_kwargs = dict(features_extractor_class=Robot1DFeatureExtractor)

        if self.robot_level == 0:
            # fmt: off
            model = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=2)
        else:
            print("loading model ...")
            model = PPO.load(self.previous_save_path, self.env)

        # fmt: off
        model.learn(total_timesteps=int(1e9), reset_num_timesteps=False,
                    callback=RobotCallback(verbose=0, max_steps=self.MAX_STEPS_FOR_ROBOT_EPISODE))
        
        print("saving model ...")
        model_save_path = f"saved_models/robot/model_{int(time())}_{self.robot_level}"
        self.previous_save_path = model_save_path
        model.save(model_save_path)
        
        self.terminal_state_flag = self.env.success_flag and (self.current_difficulty >= self.desired_difficulty)
        reward = self.__get_reward()

        if self.current_difficulty >= self.desired_difficulty:
            self.done = True
            self.episodes += 1

        self.time_steps += 1
        
        # Flag to advance to next level
        advance_flag = uniform(0, 1) <= self.ADVANCE_PROBABILITY
        self.robot_level = (self.robot_level + advance_flag) * advance_flag

        return self._make_obs(), reward, self.done, {"episodes_count": self.episodes}

    def render(self):
        pass

    def _make_obs(self):
        """Create observations

        Returns:
            List: observation vector
        """
        return [self.robot_level, self.current_robot_reward, self.current_difficulty]

    def _convert_action_to_dict_format(self, action):
        """Convert action form list format to dict format

        Args:
            action (list): output of planner model

        Returns:
            dict: action dictionay (robotPosition, goalPosition, numberOfObstacles)
        """
        # TODO: Find out why is the models stop being saved at Agent_Model_416
        planner_output = {}
        action[0] = max(action[0], 0.1)
        action[0] = min(action[0], 0.9)
        action[1] = max(action[1], 0.1)
        action[1] = min(action[1], 0.9)

        for i in range(len(action)):
            planner_output["{}".format(self.action_space_names[i])] = action[i]
        print("teacher_action = {}".format(planner_output))
        return planner_output

    def _get_robot_position_from_action(self, action: dict) -> Tuple:
        """Clip robot/ goal positions

        Args:
            action (dict): action dict from model

        Returns:
            Tuple: clipped positions
        """
        px = np.clip(
            RobotEnv.WIDTH * action["robot_position"], a_min=0, a_max=RobotEnv.WIDTH - 2
        )  # type: ignore
        py = np.clip(
            RobotEnv.HEIGHT * action["robot_position"],
            a_min=0,
            a_max=RobotEnv.HEIGHT - 2,
        )
        gx = np.clip(
            RobotEnv.WIDTH * action["goal_position"], a_min=0, a_max=RobotEnv.WIDTH - 2
        )
        gy = np.clip(
            RobotEnv.HEIGHT * action["goal_position"],
            a_min=0,
            a_max=RobotEnv.HEIGHT - 2,
        )
        return px, py, gx, gy

    def __get_reward(self) -> float:
        """Calculate current reward

        Returns:
            float: current reward
        """
        reward = (
            (self.current_difficulty / self.desired_difficulty)
            * (self.current_robot_reward / self.MAX_REWARD)
        ) ** self.ALPHA + self.terminal_state_flag * (
            1 - self.time_steps / self.MAX_TIMESTEPS
        ) * self.TERMINAL_STATE_REWARD
        return reward

    def __generate_obstacles_points(self, obstacles_count: int) -> None:
        """Generate obstacles based on teacher action for next robot session

        Args:
            obstacles_count (int): number of obstacles
        """
        for i in range(int(obstacles_count)):
            overlap = True
            new_obstacle = SingleObstacle()
            while overlap:
                px = randint(0, RobotEnv.WIDTH)
                py = randint(0, RobotEnv.HEIGHT)
                new_width = randint(50, 500)
                new_height = randint(50, 500)
                new_obstacle = SingleObstacle(px, py, new_width, new_height)
                overlap = self.env.robot.is_overlapped(new_obstacle)
            self.env.obstacles += new_obstacle

    def reset(self):
        self.time_steps = 0
        self.done = 0
        return self._make_obs()


if __name__ == "__main__":
    planner_env = TeacherEnv()
    policy_kwargs = {
        "features_extractor_class": CustomLSTM,
        "features_extractor_kwargs": dict(features_dim=2),
    }
    model = PPO(CustomActorCriticPolicy, planner_env, verbose=1)
    model.learn(total_timesteps=int(1e7))
    model.save(f"saved_models/teacher/model_{int(time())}")
