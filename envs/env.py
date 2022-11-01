from typing import Any, List
from gym import Env, spaces
from utils.action import *
from obstacle.obstacles import Obstacles
import numpy as np
from CMap2D import (
    flatten_contours,
    render_contours_in_lidar,
    CMap2D,
)
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from utils.calculations import *
import threading
from utils.robot import Robot
from obstacle.single_obstacle import SingleObstacle
from utils.lidar_rings import LidarRings

from utils.planner_checker import PlannerChecker
import random
from PIL import Image


ROBOT_RADIUS = 20


class RobotEnv(Env):
    COLLISION_SCORE = -25
    REACHED_GOAL_SCORE = 100
    EPSILON = 0.1
    WIDTH = 1280
    HEIGHT = 720
    DELTA_T = 1
    MINIMUM_VELOCITY = 0.1
    MINIMUM_DISTANCE = 0.1
    MAXIMUM_DISTANCE = 1470
    VELOCITY_STD = 2.0
    ALPHA = 0.4
    RENDER_EACH = 1
    GREEN_COLOR = (50, 225, 30)
    BLUE_COLOR = (0, 0, 255)
    BLACK_COLOR = (0, 0, 0)
    WHITE_COLOR = (255, 255, 255, 255)
    bgcolor = np.array([0.4, 0.8, 0.4])
    obstcolor = np.array([0.3, 0.3, 0.3])
    goalcolor = np.array([1.0, 1.0, 0.3])
    goallinecolor = 0.9 * bgcolor
    nosecolor = np.array([0.3, 0.3, 0.3])
    agentcolor = np.array([0.0, 1.0, 1.0])

    def __init__(self, resizable=True) -> None:
        super(RobotEnv, self).__init__()
        self.action_space_names = ["ActionXY", "ActionRot"]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
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
        self.n_angles = 1080
        self.lidarAngleIncrement = 0.005817764  # = 1/3 degrees
        self.lidarMinAngle = 0
        self.lidarMaxAngle = 6.277367543 + self.lidarAngleIncrement  # = 2*pi
        self.lidarScan = None
        self.converterCMap2D = CMap2D()
        self.converterCMap2D.set_resolution(1.0)
        self.viewer = None
        self.reward = 0
        self.planner_output = {}
        self.total_reward = 0
        self.success_flag = False

        """steps the agent took during the current episode.
        re-initialzed between episodes.
        """
        self.current_episode_timesteps = 0

        """max allowed timesteps the agent is allowed to take 
        at each episode (should be set at the planner code)."""
        self.max_episode_timesteps = 1e6

        self.done = False

        self.robot_initial_px = 0
        self.robot_initial_py = 0
        self.robot_goal_px = 0
        self.robot_goal_py = 0

        self.is_initial_state = True
        self.actually_done = False

    def step(self, action: List):
        """Step into the new state using an action given by the agent model

        Args:
            action (list): velocity action (vx, vy) provided by the agent model

        Returns:
            dict : environment state as the agent observation
        """
        self.reward = 0
        self.actually_done = False
        # If something happens, don't do another step
        if self.done:
            return self._make_obs(), self.reward, self.done, {"episode": 1}

        self.current_episode_timesteps += 1

        new_action = self._convert_action_to_ActionXY_format(action)

        self.robot.step(new_action, self.DELTA_T)

        self.reward = self.__get_reward()

        if self.current_episode_timesteps % self.RENDER_EACH == 0:
            self.render()

        self.total_reward += self.reward
        self.actually_done = self.done
        return self._make_obs(), self.reward, self.done, {}

    def __get_reward(self) -> float:
        """Calculate current reward

        Returns:
            float: reward
        """
        distance_to_goal = point_to_point_distance(
            (self.robot.px, self.robot.py), (self.robot.gx, self.robot.gy)
        )
        vx, vy = self.robot.vx, self.robot.vy
        velolicty = (vx**2 + vy**2) ** 0.5
        dist_ratio = distance_to_goal / self.MAXIMUM_DISTANCE

        reward = (1 - dist_ratio**self.ALPHA) * (
            1 - max(velolicty / self.VELOCITY_STD, self.MINIMUM_VELOCITY)
        ) ** (1 / max(dist_ratio, self.MINIMUM_DISTANCE))

        # if collision detected, add -100 and raise done flag
        if self.detect_collison():
            print("|--collision detected--|")
            reward += self.COLLISION_SCORE
            self.done = True
            self.success_flag = False
            return reward

        # if reached goal, add 1800 and raise sucess/done flags
        if self.robot.reached_destination():
            reward += self.REACHED_GOAL_SCORE
            self.done = True
            self.success_flag = True
            return reward

        if self.current_episode_timesteps >= self.max_episode_timesteps:
            self.done = True
            self.success_flag = False

        return reward

    def set_robot_position(self, px: float, py: float, gx: float, gy: float) -> None:
        """Initialize robot/goal position
        Should be called from teacher

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

    def add_boarder_obstacles(self):
        # fmt: off
        self.obstacles = Obstacles([
                SingleObstacle(-self.EPSILON, 0, self.EPSILON, self.HEIGHT),  # left obstacle
                SingleObstacle(0, -self.EPSILON, self.WIDTH, self.EPSILON),  # bottom obstacle
                SingleObstacle(self.WIDTH, 0, self.EPSILON, self.HEIGHT),  # right obstacle
                SingleObstacle(0, self.HEIGHT, self.WIDTH, self.EPSILON),  # top obstacle
        ])

    def generate_obstacles_points(self) -> List:
        """Get obstacle points as flattened contours

        Returns:
            list: contours of env obstacles
        """
        self.contours = []
        for obstacle in self.obstacles.obstacles_list:
            self.contours.append(obstacle.get_points())
        self.flat_contours = flatten_contours(self.contours)
        return self.contours

    def _make_obs(self):
        """Create agent observation from environment state and LiDAR

        Returns:
            dict: agent observation
        """
        robot = self.robot
        lidar_pos = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        ranges = np.ones((self.n_angles,), dtype=np.float32) * 25.0
        angles = (
            np.linspace(
                self.lidarMinAngle,
                self.lidarMaxAngle - self.lidarAngleIncrement,
                self.n_angles,
            )
            + lidar_pos[2]
        )
        self.generate_obstacles_points()
        render_contours_in_lidar(ranges, angles, self.flat_contours, lidar_pos[:2])
        self.lidar_scan = ranges
        self.lidar_angles = angles

        baselink_in_world = np.array([robot.px, robot.py, robot.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)

        robotvel_in_world = np.array([robot.vx, robot.vy, robot.w])
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink])
        obs = {"lidar": self.lidar_scan , "robot": robotstate_obs}

        return obs

    def render(
        self, close: Any = False, save_to_file: Any = False, show_score: Any = True
    ):
        """Render robot and obstacles on an openGL window using gym viewer

        Args:
            close (bool, optional): flag to close the environment window. Defaults to False.
            save_to_file (bool, optional): flag to save render data to a file. Defaults to False.
            show_score (boo, optional): flag to show reward on window. Defaults to True.

        Returns:
            bool: flag to check the status of the openGL window
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        WINDOW_W = self.WIDTH
        WINDOW_H = self.HEIGHT
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
                color=self.WHITE_COLOR,
            )
            self.iteration_label = pyglet.text.Label(
                "0000",
                font_size=12,
                x=20,
                y=int((WINDOW_H * 1.6) // 40.00),
                anchor_x="left",
                anchor_y="center",
                color=self.WHITE_COLOR,
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
            gl.glColor4f(self.bgcolor[0], self.bgcolor[1], self.bgcolor[2], 1.0)
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
            self.obstacle_vertices = self.generate_obstacles_points()
            for poly in self.obstacle_vertices:
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(self.obstcolor[0], self.obstcolor[1], self.obstcolor[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
            # LIDAR
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
                    color = self.agentcolor
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
                gl.glColor4f(self.nosecolor[0], self.nosecolor[1], self.nosecolor[2], 1)
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
            gl.glColor4f(self.goalcolor[0], self.goalcolor[1], self.goalcolor[2], 1)
            triangle = make_circle((xgoal, ygoal), r)
            for vert in triangle:
                gl.glVertex3f(vert[0], vert[1], 0)
            gl.glEnd()
            # Goal line
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor4f(
                self.goallinecolor[0], self.goallinecolor[1], self.goallinecolor[2], 1
            )
            gl.glVertex3f(rx, ry, 0)
            gl.glVertex3f(xgoal, ygoal, 0)
            gl.glEnd()
            # --
            self.transform.disable()

            self.score_label.text = ""
            if show_score:
                self.score_label.text = "R {:0.4f}".format(self.reward)
                self.iteration_label.text = "iter {}".format(
                    self.current_episode_timesteps
                )
            self.score_label.draw()
            self.iteration_label.draw()
            win.flip()
            if save_to_file:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    "/tmp/navreptrainenv{:05}.png".format(
                        self.current_episode_timesteps
                    )
                )
            return self.viewer.isopen

    def detect_collison(self):
        """Detect if the agent has collided with any obstacle

        Returns:
            bool: flag to check collisions
        """
        collision_flag = False
        for obstacle in self.obstacles.obstacles_list:
            collision_flag |= self.robot.is_overlapped(obstacle=obstacle)
        return collision_flag

    def _convert_action_to_ActionXY_format(self, action: List):
        """Convert action array into action object

        Args:
            action (list): list of velocities given by agent model

        Returns:
            ActionXY : same action by given in object format
        """
        real_angle = action[2] * np.pi  # -1, 1 -> -pi, pi
        return ActionXY(action[0], action[1], 0)

    def reset(self):
        """
        Reset robot state and generate new obstacles points
        Returns:
            dict: observation of the current environment state
        """
        if self.done or self.is_initial_state:
            print("resting robot env ...")
            self.robot.set_position([self.robot_initial_px, self.robot_initial_py])
            self.robot.set_goal_position([self.robot_goal_px, self.robot_goal_py])
            self.add_boarder_obstacles()

            self.done = False
            self.success_flag = False
            self.is_initial_state = False
            self.current_episode_timesteps = 0

        self.generate_obstacles_points()
        return self._make_obs()
