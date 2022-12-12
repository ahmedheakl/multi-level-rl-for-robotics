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
max_episode_steps = 1000
# 1e5
max_session_steps = 10000

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
progress_discount = 0.4

[render]
render_each = 200
save_to_file = False

[env]
epsilon = 1

[statistics]
collect_statistics = True
scenario = train
"""

# ---------------------TEAHCER CONFIGURATIONS-----------------#
teacher_config_str = """
[reward]
alpha = 0.4
terminal_state_reward = 100
max_reward = 3600
base_difficulty = 590
overlap_goal_penality = -100
infinite_difficulty_penality = -100
too_close_to_goal_penality = -50
is_goal_or_robot_overlap_obstacles_penality = -100
gamma = 0.4
diff_increase_factor = 1.15
base_num_successes = 5
num_successes_increase_factor = 1.2

[env]
advance_probability = 0.9
max_hard_obstacles_count = 2
max_medium_obstacles_count = 5
max_small_obstacles_count = 7
hard_obstacles_max_dim = 300
hard_obstacles_min_dim = 200
medium_obstacles_max_dim = 200
medium_obstacles_min_dim = 100
small_obstacles_max_dim = 100
small_obstacles_min_dim = 50
# {flat: 1D, rings: 2D}
lidar_mode = flat

[render]
render_eval = False

[statistics]
scenario = train
collect_statistics = True
robot_log_eval_freq = 100
n_robot_eval_episodes = 0
save_model_freq = 1

[timesteps]
max_sessions = 5
"""

# ---------------------EVALUATIONS CONFIGURATIONS-----------------#
eval_config_str = """
[dimensions]
width = 1080
height = 720
robot_radius = 20
goal_radius = 10

[positions]
robot_initial_px = 50
robot_initial_py = 50
robot_goal_px = 900
robot_goal_py = 700

[obstacles]
n_hard = 5
n_,medium = 7
n_small = 9

[timesteps]
delta_t = 1
# 1e3
max_episode_steps = 3000
# 1e5
max_session_steps = 300000

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
progress_discount = 0.4

[render]
render_each = 1
save_to_file = False

[env]
epsilon = 1

[statistics]
collect_statistics = True
scenario = train
"""
