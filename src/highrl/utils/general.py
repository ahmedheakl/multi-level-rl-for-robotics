"""Utilties for HighRL"""
from dataclasses import dataclass
from configparser import RawConfigParser


@dataclass
class TeacherConfigs:
    """Configurations for the teacher"""

    max_robot_episode_steps: int
    max_session_timesteps: int
    alpha: float
    terminal_state_reward: int
    max_robot_episode_reward: int
    base_difficulty: int
    overlap_goal_penality: int
    infinite_difficulty_penality: int
    too_close_to_goal_penality: int
    is_goal_or_robot_overlap_obstacles_penality: int
    gamma: float
    diff_increase_factor: float
    base_num_successes: int
    num_successes_increase_factor: float
    advance_probability: float
    max_big_obstacles_count: int
    max_med_obstacles_count: int
    max_small_obstacles_count: int
    big_obstacles_max_dim: int
    big_obstacles_min_dim: int
    med_obstacles_max_dim: int
    med_obstacles_min_dim: int
    small_obstacles_max_dim: int
    small_obstacles_min_dim: int
    lidar_mode: str
    collect_statistics: bool
    scenario: str
    robot_log_eval_freq: int
    teacher_save_model_freq: int
    n_robot_eval_episodes: int
    render_eval: bool

    def compute_success(self, episodes: int) -> int:
        """Calculate the number of success"""
        num_sucesses = int((self.num_successes_increase_factor) ** episodes)
        num_sucesses *= self.base_num_successes
        return num_sucesses

    @property
    def max_obstacles_count(self) -> int:
        """Retrieve the sum of obstacles of all sizes"""
        return (
            self.max_big_obstacles_count
            + self.max_med_obstacles_count
            + self.max_small_obstacles_count
        )


def configure_teacher(config: RawConfigParser) -> TeacherConfigs:
    """Configure the environment variables using input config object
    Args:
        config (RawConfigParser): input config object
    """
    cfg = TeacherConfigs(
        max_robot_episode_steps=config.getint("timesteps", "max_episode_timesteps"),
        max_session_timesteps=config.getint("timesteps", "max_session_timesteps"),
        alpha=config.getfloat("reward", "alpha"),
        terminal_state_reward=config.getint("reward", "terminal_state_reward"),
        max_robot_episode_reward=config.getint("reward", "max_reward"),
        base_difficulty=config.getint("reward", "base_difficulty"),
        overlap_goal_penality=config.getint("reward", "overlap_goal_penality"),
        infinite_difficulty_penality=config.getint(
            "reward", "infinite_difficulty_penality"
        ),
        too_close_to_goal_penality=config.getint(
            "reward", "too_close_to_goal_penality"
        ),
        is_goal_or_robot_overlap_obstacles_penality=config.getint(
            "reward", "is_goal_or_robot_overlap_obstacles_penality"
        ),
        gamma=config.getfloat("reward", "gamma"),
        diff_increase_factor=config.getfloat("reward", "diff_increase_factor"),
        base_num_successes=config.getint("reward", "base_num_successes"),
        num_successes_increase_factor=config.getfloat(
            "reward", "num_successes_increase_factor"
        ),
        advance_probability=config.getfloat("env", "advance_probability"),
        max_big_obstacles_count=config.getint("env", "max_hard_obstacles_count"),
        max_med_obstacles_count=config.getint("env", "max_medium_obstacles_count"),
        max_small_obstacles_count=config.getint("env", "max_small_obstacles_count"),
        big_obstacles_min_dim=config.getint("env", "hard_obstacles_min_dim"),
        big_obstacles_max_dim=config.getint("env", "hard_obstacles_max_dim"),
        med_obstacles_min_dim=config.getint("env", "medium_obstacles_min_dim"),
        med_obstacles_max_dim=config.getint("env", "medium_obstacles_max_dim"),
        small_obstacles_min_dim=config.getint("env", "small_obstacles_min_dim"),
        small_obstacles_max_dim=config.getint("env", "small_obstacles_max_dim"),
        lidar_mode=config.get("env", "lidar_mode"),
        collect_statistics=config.getboolean("statistics", "collect_statistics"),
        scenario=config.get("statistics", "scenario"),
        robot_log_eval_freq=config.getint("statistics", "robot_log_eval_freq"),
        teacher_save_model_freq=config.getint("statistics", "save_model_freq"),
        n_robot_eval_episodes=config.getint("statistics", "n_robot_eval_episodes"),
        render_eval=config.getboolean("render", "render_eval"),
    )
    return cfg


@dataclass
class RobotConfigs:
    """Configurations for the robot training"""

    width: int
    height: int
    robot_radius: int
    goal_radius: int

    robot_init_x_pos: int
    robot_init_y_pos: int
    goal_x_pos: int
    goal_y_pos: int

    eval_big_obs_count: int
    eval_med_obs_count: int
    eval_sml_obs_count: int

    eval_big_obs_dim: int
    eval_med_obs_dim: int
    eval_sml_obs_dim: int

    delta_t: float
    max_episode_steps: int

    n_angles: int
    lidar_angle_increment: float
    lidar_min_angle: float
    lidar_max_angle: float

    collision_score: int
    reached_goal_score: int
    minimum_velocity: float
    minimum_distance: float
    maximum_distance: float
    velocity_std: float
    alpha: float
    progress_discount: float

    render_each: int
    save_to_file: bool

    epsilon: int
    collect_statistics: bool
    scenario: str

    env_render_path: str


def configure_robot(config: RawConfigParser, env_render_path: str) -> RobotConfigs:
    """Configure environment variables using input config object

    Args:
        config (RawConfigParser): input config object
    """
    return RobotConfigs(
        width=config.getint("dimensions", "width"),
        height=config.getint("dimensions", "height"),
        robot_radius=config.getint("dimensions", "robot_radius"),
        goal_radius=config.getint("dimensions", "goal_radius"),
        delta_t=config.getfloat("timesteps", "delta_t"),
        max_episode_steps=config.getint("timesteps", "max_episode_steps"),
        n_angles=config.getint("lidar", "n_angles"),
        lidar_angle_increment=config.getfloat("lidar", "lidar_angle_increment"),
        lidar_min_angle=config.getfloat("lidar", "lidar_min_angle"),
        lidar_max_angle=config.getfloat("lidar", "lidar_max_angle"),
        collision_score=config.getint("reward", "collision_score"),
        reached_goal_score=config.getint("reward", "reached_goal_score"),
        minimum_velocity=config.getfloat("reward", "minimum_velocity"),
        minimum_distance=config.getfloat("reward", "minimum_distance"),
        maximum_distance=config.getfloat("reward", "maximum_distance"),
        velocity_std=config.getfloat("reward", "velocity_std"),
        alpha=config.getfloat("reward", "alpha"),
        progress_discount=config.getfloat("reward", "progress_discount"),
        render_each=config.getint("render", "render_each"),
        save_to_file=config.getboolean("render", "save_to_file"),
        epsilon=config.getint("env", "epsilon"),
        collect_statistics=config.getboolean("statistics", "collect_statistics"),
        scenario=config.get("statistics", "scenario"),
        env_render_path=env_render_path,
        robot_init_x_pos=config.getint("eval", "robot_init_x_pos"),
        robot_init_y_pos=config.getint("eval", "robot_init_y_pos"),
        goal_x_pos=config.getint("eval", "goal_x_pos"),
        goal_y_pos=config.getint("eval", "goal_y_pos"),
        eval_big_obs_count=config.getint("eval", "eval_big_obs_count"),
        eval_med_obs_count=config.getint("eval", "eval_med_obs_count"),
        eval_sml_obs_count=config.getint("eval", "eval_sml_obs_count"),
        eval_big_obs_dim=config.getint("eval", "eval_big_obs_dim"),
        eval_med_obs_dim=config.getint("eval", "eval_med_obs_dim"),
        eval_sml_obs_dim=config.getint("eval", "eval_sml_obs_dim"),
    )
