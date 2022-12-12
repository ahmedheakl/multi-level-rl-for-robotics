from typing import Any, List
from gym import Env, spaces
from highrl.utils.action import *
from highrl.obstacle.obstacles import Obstacles
import numpy as np
from CMap2D import render_contours_in_lidar

from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from highrl.utils.calculations import *
import threading
from highrl.agents.robot import Robot
from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.configs.colors import *
import configparser
import pandas as pd
import time
import argparse
from os import path, mkdir
import time


class RobotEnv(Env):
    def __init__(
        self, config: configparser.RawConfigParser, args: argparse.Namespace
    ) -> None:
        super(RobotEnv, self).__init__()
        self.action_space_names = ["ActionXY", "ActionRot"]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32
                ),
                "robot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                ),
            }
        )

        self.obstacles = Obstacles()
        self.robot = Robot()
        self.viewer = None
        self.args = args
        self.reward = 0
        self.episodes = 1
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_reward = 0
        self.total_steps = 0
        self.success_flag = False
        self.num_successes = 0

        self.done = False

        self.robot_initial_px = 0
        self.robot_initial_py = 0
        self.robot_goal_px = 0
        self.robot_goal_py = 0

        self.is_initial_state = True

        """Results of each episode
        Contains [episode_reward, episode_steps, success_flag]
        """
        self.results = []

        self._configure(config=config)

        self.episode_statistics = None
        if self.collect_statistics:
            self.episode_statistics = pd.DataFrame(
                columns=[
                    "total_steps",
                    "episode_steps",
                    "scenario",
                    "damage",
                    "goal_reached",
                    "total_reward",
                    "episode_reward",
                    "reward",
                    "wall_time",
                ]
            )

    def _configure(self, config: configparser.RawConfigParser) -> None:
        """Configure environment variables using input config object

        Args:
            config (configparser.RawConfigParser): input config object
        """
        self.config = config

        self.width = config.getint("dimensions", "width")
        self.height = config.getint("dimensions", "height")
        self.robot_radius = config.getint("dimensions", "robot_radius")
        self.goal_radius = config.getint("dimensions", "goal_radius")

        self.delta_t = config.getfloat("timesteps", "delta_t")
        self.max_episode_steps = config.getint("timesteps", "max_episode_steps")

        self.n_angles = config.getint("lidar", "n_angles")
        self.lidar_angle_increment = config.getfloat("lidar", "lidar_angle_increment")
        self.lidar_min_angle = config.getfloat("lidar", "lidar_min_angle")
        self.lidar_max_angle = config.getfloat("lidar", "lidar_max_angle")

        self.collision_score = config.getint("reward", "collision_score")
        self.reached_goal_score = config.getint("reward", "reached_goal_score")
        self.minimum_velocity = config.getfloat("reward", "minimum_velocity")
        self.minimum_distance = config.getfloat("reward", "minimum_distance")
        self.maximum_distance = config.getfloat("reward", "maximum_distance")
        self.velocity_std = config.getfloat("reward", "velocity_std")
        self.alpha = config.getfloat("reward", "alpha")
        self.progress_discount = config.getfloat("reward", "progress_discount")

        self.render_each = config.getint("render", "render_each")
        self.save_to_file = config.getboolean("render", "save_to_file")

        self.epsilon = config.getint("env", "epsilon")

        self.collect_statistics = config.getboolean("statistics", "collect_statistics")
        self.scenario = config.get("statistics", "scenario")

    def step(self, action: List) -> Tuple:
        """Step into a new state using an action given by the robot model

        Args:
            action (List): velocity action (vx, vy) provided by the robot model

        Returns:
            Tuple : observation, reward, done, info
        """
        self.reward = 0
        self.episode_steps += 1
        self.total_steps += 1

        new_action = self._convert_action_to_ActionXY_format(action)

        old_distance_to_goal = point_to_point_distance(
            (self.robot.px, self.robot.py), (self.robot.gx, self.robot.gy)
        )
        self.robot.step(new_action, self.delta_t)
        new_distance_to_goal = point_to_point_distance(
            (self.robot.px, self.robot.py), (self.robot.gx, self.robot.gy)
        )

        self.reward = (
            self.__get_reward()
            + (old_distance_to_goal - new_distance_to_goal) * self.progress_discount
        )

        self.episode_reward += self.reward

        if self.episode_steps % self.render_each == 0:
            self.render(save_to_file=self.save_to_file)

        if self.collect_statistics:
            self.episode_statistics.loc[len(self.episode_statistics)] = [  # type: ignore
                self.total_steps,
                self.episode_steps,
                "robot_env_" + self.scenario,
                100 if self.detect_collison() else 0,
                self.robot.reached_destination(),
                self.total_reward,
                self.episode_reward,
                self.reward,
                time.time(),
            ]
        # log data
        if self.done:
            self.num_successes += self.success_flag
            self.results.append(
                [self.episode_reward, self.episode_steps, self.success_flag]
            )

        return self._make_obs(), self.reward, self.done, {}

    def __get_reward(self) -> float:
        """Calculates current reward

        Returns:
            float: current reward value
        """
        reward = 0.0
        if self.detect_collison():
            print("|--collision detected--|")
            reward += self.collision_score
            self.done = True
            self.success_flag = False
            return reward

        if self.robot.reached_destination():
            reward += self.reached_goal_score
            self.done = True
            self.success_flag = True
            return reward

        if self.episode_steps >= self.max_episode_steps:
            self.done = True
            self.success_flag = False
            self.episodes += 1

        return reward

    def set_robot_position(self, px: float, py: float, gx: float, gy: float) -> None:
        """Initializes robot and goal positions
        Should be called from ``teacher``

        Args:
            px (float): x_position of robot
            py (float): y_position of robot
            gx (float): x_position of goal
            py (float): y_position of goal
        """
        self.robot_initial_px = px
        self.robot_initial_py = py
        self.robot_goal_px = gx
        self.robot_goal_py = gy

        self.robot.set_position([px, py])
        self.robot.set_goal_position([gx, gy])

    def add_boarder_obstacles(self) -> None:
        """Creates border obstacles to limit the allowable navigation area"""
        # fmt: off
        self.obstacles = Obstacles([
                SingleObstacle(-self.epsilon, 0, self.epsilon, self.height),  # left obstacle
                SingleObstacle(0, -self.epsilon, self.width, self.epsilon),  # bottom obstacle
                SingleObstacle(self.width, 0, self.epsilon, self.height),  # right obstacle
                SingleObstacle(0, self.height, self.width, self.epsilon),  # top obstacle
        ])

    def _make_obs(self) -> dict:
        """Creates robot observation from environment state and LiDAR

        Returns:
            dict: robot observation
        """
        robot = self.robot
        lidar_pos = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        ranges = np.ones((self.n_angles,), dtype=np.float32) * 25.0
        angles = (
            np.linspace(
                self.lidar_min_angle,
                self.lidar_max_angle - self.lidar_angle_increment,
                self.n_angles,
            )
            + lidar_pos[2]
        )
        render_contours_in_lidar(ranges, angles, self.flat_contours, lidar_pos[:2])

        self.lidar_scan = ranges
        self.lidar_angles = angles

        baselink_in_world = np.array([robot.px, robot.py, robot.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        robotvel_in_world = np.array([robot.vx, robot.vy, 0])
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink])

        return {"lidar": self.lidar_scan, "robot": robotstate_obs}

    def render(
        self, close: bool = False, save_to_file: bool = False, show_score: bool = True
    ) -> bool:
        """Renders robot and obstacles on an openGL window using gym viewer

        Args:
            close (bool, optional): flag to close the environment window. Defaults to False.
            save_to_file (bool, optional): flag to save render data to a file. Defaults to False.
            show_score (bool, optional): flag to show reward on window. Defaults to True.

        Returns:
            bool: flag to check the status of the openGL window
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        WINDOW_W = self.width
        WINDOW_H = self.height
        VP_W = WINDOW_W
        VP_H = WINDOW_H
        from gym.envs.classic_control import rendering
        import pyglet
        from pyglet import gl

        # Create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()
            self.transform.set_scale(10, 10)
            self.transform.set_translation(128, 128)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=12,
                x=20,
                y=int(WINDOW_H * 2.5 / 40.00),
                anchor_x="left",
                anchor_y="center",
                color=white_color,
            )
            self.iteration_label = pyglet.text.Label(
                "0000",
                font_size=12,
                x=20,
                y=int((WINDOW_H * 1.6) // 40.00),
                anchor_x="left",
                anchor_y="center",
                color=white_color,
            )
            self.transform = rendering.Transform()
            self.image_lock = threading.Lock()

        def make_circle(c, r, res=10):
            """Create circle points

            Args:
                c (list): center of the circle
                r (float): radius of the circle
                res (int, optional): resolution of points. Defaults to 10.

            Returns:
                list: vertices representing with desired resolution
            """
            thetas = np.linspace(0, 2 * np.pi, res + 1)[:-1]
            verts = np.zeros((res, 2))
            verts[:, 0] = c[0] + r * np.cos(thetas)
            verts[:, 1] = c[1] + r * np.sin(thetas)
            return verts

        with self.image_lock:
            self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()
            win.clear()
            gl.glViewport(0, 0, VP_W, VP_H)
            # Green background
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
            gl.glVertex3f(0, VP_H, 0)
            gl.glVertex3f(VP_W, VP_H, 0)
            gl.glVertex3f(VP_W, 0, 0)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            # Transform
            rx = self.robot.px
            ry = self.robot.py
            rt = self.robot.theta
            self.transform.enable()  # applies T_sim_in_viewport to below coords (all in sim frame)
            # Map closed obstacles ---
            self.obstacle_vertices = self.contours
            for poly in self.obstacle_vertices:
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
            # TODO: show lidar
            # Agent body
            for n, agent in enumerate([self.robot]):
                px = agent.px
                py = agent.py
                angle = self.robot.fix(agent.theta + np.pi / 2, 2 * np.pi)
                r = agent.radius
                # Agent as Circle
                poly = make_circle((px, py), r)
                gl.glBegin(gl.GL_POLYGON)
                if n == 0:
                    color = np.array([1.0, 1.0, 1.0])
                else:
                    color = agentcolor
                gl.glColor4f(color[0], color[1], color[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Direction triangle
                xnose = px + r * np.cos(angle)
                ynose = py + r * np.sin(angle)
                xright = px + 0.3 * r * -np.sin(angle)
                yright = py + 0.3 * r * np.cos(angle)
                xleft = px - 0.3 * r * -np.sin(angle)
                yleft = py - 0.3 * r * np.cos(angle)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                gl.glVertex3f(xnose, ynose, 0)
                gl.glVertex3f(xright, yright, 0)
                gl.glVertex3f(xleft, yleft, 0)
                gl.glEnd()
            # Goal
            xgoal = self.robot.gx
            ygoal = self.robot.gy
            r = self.robot.goal_radius

            # Goal markers
            gl.glBegin(gl.GL_POLYGON)
            gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
            triangle = make_circle((xgoal, ygoal), r)
            for vert in triangle:
                gl.glVertex3f(vert[0], vert[1], 0)
            gl.glEnd()
            # Goal line
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor4f(goallinecolor[0], goallinecolor[1], goallinecolor[2], 1)
            gl.glVertex3f(rx, ry, 0)
            gl.glVertex3f(xgoal, ygoal, 0)
            gl.glEnd()
            # --
            self.transform.disable()

            self.score_label.text = ""
            if show_score:
                self.score_label.text = "R {:0.4f}".format(self.reward)
                self.iteration_label.text = "iter {}".format(self.episode_steps)
            self.score_label.draw()
            self.iteration_label.draw()
            win.flip()
            if save_to_file:
                save_folder = path.join(self.args.env_render_path, f"{self.episodes}")
                if not path.isdir(save_folder):
                    mkdir(save_folder)
                save_path = path.join(
                    save_folder, "{:08}.png".format(self.episode_steps)
                )
                pyglet.image.get_buffer_manager().get_color_buffer().save(save_path)
            return self.viewer.isopen

    def detect_collison(self) -> bool:
        """Detects if the robot has collided with any obstacles

        Returns:
            bool: flag to check collisions. Ouputs True if there is collision
        """
        collision_flag = False
        for obstacle in self.obstacles.obstacles_list:
            collision_flag |= self.robot.is_overlapped(obstacle=obstacle)
        return collision_flag

    def _convert_action_to_ActionXY_format(self, action: List) -> ActionXY:
        """Converts action array into action object

        Args:
            action (List): list of velocities representing the robot action

        Returns:
            ActionXY: same action but given in object format
        """
        return ActionXY(action[0], action[1], 0)

    def reset(self) -> dict:
        """Resets robot state and generate new obstacles points

        Returns:
            dict: observation of the current environment state
        """
        if self.done or self.is_initial_state:
            print("reseting robot env ...")
            self.robot.set_position([self.robot_initial_px, self.robot_initial_py])
            self.robot.set_goal_position([self.robot_goal_px, self.robot_goal_py])
            self.total_reward += self.episode_reward
            if self.is_initial_state:
                self.results = []
                self.total_reward = 0
                self.total_steps = 0

            self.success_flag = False
            self.is_initial_state = False
            self.episode_steps = 0
            self.done = False
            self.episode_reward = 0
            self.flat_contours, self.contours = self.obstacles.get_flatten_contours()
        return self._make_obs()
