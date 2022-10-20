import numpy as np

# Policies
# from https://github.com/vita-epfl/CrowdNav/tree/master/crowd_sim/envs/policy
from policy.linear_policy import Linear
from policy.orca import ORCA


#from https://github.com/mit-acl/gym-collision-avoidance/tree/6d008b93e9e2c68fdd7b4b867c63210777b6aea8/gym_collision_avoidance/envs/policies

from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
# from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from gym_collision_avoidance.envs.policies.LearningPolicyGA3C import LearningPolicyGA3C


def none_policy():
    return None




policy_dict = {
    'linear' : Linear,
    'orca' : ORCA,
    'RVO': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    'learning': LearningPolicy,
    'learning_ga3c': LearningPolicyGA3C,
    'static': StaticPolicy,
    'CADRL': CADRLPolicy,
    'none' : none_policy
}

def none_policy():
    return None

