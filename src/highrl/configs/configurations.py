# ---------------------ROBOT CONFIGURATIONS-----------------#
robot_config_str = """
[dimensions]
width = 1280
height = 720
robot_radius = 20
goal_radius = 10

[timesteps]
delta_t = 1
# 1e3
max_episode_steps = 10
# 1e5
max_robot_steps = 100

[lidar]
n_angles = 1080
lidar_angle_increment = 005817764
lidar_min_angle = 0
lidar_max_angle = 6.283185307

[reward]
collision_score = -25
reached_goal_score = 100
minimum_velocity = 0.1
minimum_distance = 0.1
maximum_distance = 1470
velocity_std = 2.0
alpha = 0.4


[render]
render_each = 1

[env]
epsilon = 1
"""

# ---------------------TEAHCER CONFIGURATIONS-----------------#
teacher_config_str = """

[reward]
alpha = 0.4
terminal_state_reward = 100
max_reward = 3600
base_difficulty = 590


[env]
advance_probability = 0.9
max_obstacles_count = 10
# {flat: 1D, rings: 2D}
lidar_mode = flat
"""
