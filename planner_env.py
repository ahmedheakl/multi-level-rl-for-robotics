from typing import List
from envs.env import TestEnv
from gym import Env, spaces
from obstacle.obstacles import Obstacles
from obstacle.singleobstacle import SingleObstacle
import numpy as np
from numpy.linalg import norm
from utils.calculations import *
from policy.custom_policy import CustomFeatureExtractor
from stable_baselines3 import PPO
from random import randint
from planner.planner import CustomActorCriticPolicy
from planner.planner import CustomLSTM
from utils.planner_checker import PlannerChecker

class PlannerEnv(Env):
    def __init__(self) -> None:
        super(PlannerEnv, self).__init__()
        self.action_space_names = ["P_robot", "P_goal", "d_no.obstacles"]
        self.action_space = spaces.Box(
            low=1, high=7, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000,
                                shape=(3,), dtype=np.float32)
        self.env = TestEnv()
        self.env.reset()
        self.episodes = 0
        self.difficulty = 0
        self.planner_input = [self.env.passed, self.env.avg_success_time, self.env.dist_to_goal] 
        self.diff_checker = PlannerChecker()

    def _make_obs(self):
        """Create observation of planner

        Returns:
            list(3): obvervation list (numberOfSucesses, averageSuccessTime, distanceToGoal)
        """
        self.planner_input[0] = self.env.passed
        sum = 0
        for i in range(len(self.env.avg_success_time)):
            sum += self.env.avg_success_time[i]
        self.planner_input[1] = sum / len(self.env.avg_success_time)
        obs = self.planner_input
        return obs

    def _get_action(self, action):
        """Convert action form list format to dict format

        Args:
            action (list): output of planner model

        Returns:
            dict: action dictionay (robotPosition, goalPosition, numberOfObstacles)
        """
        #TODO: Find out why is the models stop being saved at Agent_Model_416
        planner_output = {}
        for i in range(len(action)):
            planner_output["{}".format(self.action_space_names[i])] = action[i]
        return planner_output

    

    def step(self, action):
        # TODO: dont overlap robot position
        gamma_difficulty = 10
        gamma_reward = 10
        action = self._get_action(action)
        self.env.planner_output = action
        reward = - gamma_difficulty / (self.difficulty + 1) - gamma_reward / (self.env.reward + 1)
        obst_list = []
        print(action)
        for i in range(int(action["d_no.obstacles"])):
            #TODO: check how openGL renders dims
            px = randint(0, self.env.width)
            py = randint(0,self.env.height)
            new_width = randint(50,500)
            new_height = randint(50,500)
            new_obstacle = SingleObstacle(px, py, new_width, new_height)
            
            obst_list.append(new_obstacle)
            self.env.obstacles = Obstacles(obst_list)
        
        self.env.robot.set(px=100*action["P_robot"], py=150*action["P_robot"], gx=100*action["P_goal"], gy=400*action["P_goal"],
                        gt=0, vx=0, vy=0, w=0, theta=0, radius=20)
        
        args_list = [self.env.robot.px, self.env.robot.py, self.env.robot.gx, self.env.robot.gy]
        args_list = list(map(int, args_list))
        self.difficulty = self.diff_checker.get_map_difficulity(self.env.obstacles, self.env.height, self.env.width, *args_list)
        
        


        
    # start training the mobile robot   
        policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor)
        
        if self.episodes == 0:
            model = PPO("MultiInputPolicy", self.env,
                    policy_kwargs=policy_kwargs, verbose=2)
        else:
            model = PPO.load("agent_models/Agent_Model_{}".format(self.episodes-1), self.env)
        
        
        model.learn(total_timesteps=1)
        model.save("agent_models/Agent_Model_{}".format(self.episodes))

        self.episodes += 1
        self.env.episodes += 1
        
        done = True
        
        return self._make_obs(), reward, done, {"episode_number": self.episodes}
    
    def reset(self):
        return self._make_obs()



if __name__ == '__main__':
    planner_env = PlannerEnv()
    policy_kwargs = dict(
    features_extractor_class=CustomLSTM,
    features_extractor_kwargs=dict(features_dim=2),
)
    model = PPO(CustomActorCriticPolicy, planner_env , verbose=1)
    planner_iter = 3
    for i in range(planner_iter):
        model.learn(total_timesteps=1)
        model.save("planner_models/Planner_Model_{}".format(i))
    # env = TestEnv()
    # policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor)
    # model = PPO.load("Agent_Models/Agent_Model_{}".format(416), env)
    # model.learn(total_timesteps=10)