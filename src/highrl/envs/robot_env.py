"""Implementation of Robot Environment"""

from typing import List, Tuple
import threading
import time
import argparse
from configparser import RawConfigParser
from os import path, mkdir
import numpy as np
from CMap2D import render_contours_in_lidar  # pylint: disable=no-name-in-module
from gym import Env, spaces
from gym.envs.classic_control import rendering
import pyglet
from pyglet import gl
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose

from highrl.obstacle.single_obstacle import SingleObstacle
from highrl.utils.action import ActionXY
from highrl.utils.calculations import point_to_point_distance
from highrl.agents.robot import Robot
from highrl.obstacle.obstacles import Obstacles
from highrl.utils.utils import Position, configure_robot
from highrl.configs import colors
from highrl.utils.robot_utils import RobotOpt


class RobotEnv(Env):
    """Robot Environment Class used in training and testing"""

    tensorboard_dir = "runs/robot"
    rwrd_grph_name = "reward"
    eps_rwrd_grph_name = "episode_reward"
    action_space_names = ["ActionXY", "ActionRot"]

    def __init__(
        self,
        config: RawConfigParser,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
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
        # Results of each episode
        # Contains [episode_reward, episode_steps, success_flag]
        self.results: List[Tuple[float, int, bool]] = []

        self.done = False

        self.cfg = configure_robot(config, args.env_render_path)
        self.opt = RobotOpt()
        self.opt.set_tb_writer(self.tensorboard_dir)
        self.robot.set_radius(self.cfg.robot_radius, self.cfg.goal_radius)

    def step(self, action: np.ndarray) -> Tuple:
        """Step into a new state using an action given by the robot model

        Args:
            action (List): velocity action (vx, vy) provided by the robot model

        Returns:
            Tuple : observation, reward, done, info
        """
        self.opt.reward = 0
        self.opt.episode_steps += 1
        self.opt.total_steps += 1

        new_action = self._to_actionxy_format(action)

        old_distance_to_goal = point_to_point_distance(
            (self.robot.px, self.robot.py), (self.robot.gx, self.robot.gy)
        )
        self.robot.step(new_action, self.cfg.delta_t)
        new_distance_to_goal = point_to_point_distance(
            (self.robot.px, self.robot.py), (self.robot.gx, self.robot.gy)
        )

        self.opt.reward = (
            self.__get_reward()
            + (old_distance_to_goal - new_distance_to_goal) * self.cfg.progress_discount
        )

        self.opt.episode_reward += self.opt.reward
        self.opt.tb_writer.add_scalar(
            self.rwrd_grph_name,
            self.opt.reward,
            self.opt.total_steps,
        )
        self.opt.tb_writer.add_scalar(
            self.eps_rwrd_grph_name,
            self.opt.episode_reward,
            self.opt.total_steps,
        )

        if self.opt.episode_steps % self.cfg.render_each == 0:
            self.render(save_to_file=self.cfg.save_to_file)

        if self.cfg.collect_statistics:
            self.opt.episode_statistics.append(
                [
                    self.opt.total_steps,
                    self.opt.episode_steps,
                    "robot_env_" + self.cfg.scenario,
                    100 if self.detect_collison() else 0,
                    self.robot.reached_destination(),
                    self.opt.total_reward,
                    self.opt.episode_reward,
                    self.opt.reward,
                    time.time(),
                ]
            )
        # log data
        if self.done:
            self.opt.num_successes += self.opt.success_flag
            result = (
                self.opt.episode_reward,
                self.opt.episode_steps,
                self.opt.success_flag,
            )
            self.results.append(result)

        return self._make_obs(), self.opt.reward, self.done, {}

    def __get_reward(self) -> float:
        """Calculates current reward

        Returns:
            float: current reward value
        """
        reward = 0.0
        if self.detect_collison():
            print("|--collision detected--|")
            reward += self.cfg.collision_score
            self.done = True
            self.opt.success_flag = False
            return reward

        if self.robot.reached_destination():
            reward += self.cfg.reached_goal_score
            self.done = True
            self.opt.success_flag = True
            return reward

        if self.opt.episode_steps >= self.cfg.max_episode_steps:
            self.done = True
            self.opt.success_flag = False
            self.opt.episodes += 1

        return reward

    def set_robot_position(self, robot_pos: Position, goal_pos: Position) -> None:
        """Initializes robot and goal positions
        Should be called from ``teacher``

        Args:
            robot_pos (Position): Position of the robot
            goal_pos (Position): Position of the goal
        """
        self.opt.robot_init_pos = robot_pos
        self.opt.goal_init_pos = goal_pos
        self.robot.set_position(robot_pos)
        self.robot.set_goal_position(goal_pos)

    def add_boarder_obstacles(self) -> None:
        """Creates border obstacles to limit the allowable navigation area"""
        # fmt: off
        self.obstacles = Obstacles([
                SingleObstacle(-self.cfg.epsilon, 0, self.cfg.epsilon, self.cfg.height),  # left obstacle
                SingleObstacle(0, -self.cfg.epsilon, self.cfg.width, self.cfg.epsilon),  # bottom obstacle
                SingleObstacle(self.cfg.width, 0, self.cfg.epsilon, self.cfg.height),  # right obstacle
                SingleObstacle(0, self.cfg.height, self.cfg.width, self.cfg.epsilon),  # top obstacle
        ])

    def _make_obs(self) -> dict:
        """Creates robot observation from environment state and LiDAR

        Returns:
            dict: robot observation
        """
        robot = self.robot
        lidar_pos = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        ranges = np.ones((self.cfg.n_angles,), dtype=np.float32)
        ranges.fill(25.0)
        angles = (
            np.linspace(
                self.cfg.lidar_min_angle,
                self.cfg.lidar_max_angle - self.cfg.lidar_angle_increment,
                self.cfg.n_angles,
            )
            + lidar_pos[2]
        )
        render_contours_in_lidar(ranges, angles, self.opt.flat_contours, lidar_pos[:2])

        self.opt.lidar_scan = ranges
        self.opt.lidar_angles = angles

        baselink_in_world = np.array([robot.px, robot.py, robot.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        robotvel_in_world = np.array([robot.vx, robot.vy, 0])
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink])

        return {"lidar": self.opt.lidar_scan, "robot": robotstate_obs}

    def render(
        self,
        mode="human",
        close: bool = False,
        save_to_file: bool = False,
    ) -> bool:
        """Renders robot and obstacles on an openGL window using gym viewer

        Args:
            close (bool, optional): flag to close the environment window. Defaults to False.
            save_to_file (bool, optional): flag to save render data to a file. Defaults to False.

        Returns:
            bool: flag to check the status of the openGL window
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return False

        # Create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.cfg.width, self.cfg.height)
            self.transform = rendering.Transform()
            self.transform.set_scale(10, 10)
            self.transform.set_translation(128, 128)
            self.iteration_label = pyglet.text.Label(
                "0000",
                font_size=7,
                x=20,
                y=int((self.cfg.height * 1.6) // 40.00),
                anchor_x="left",
                anchor_y="center",
                color=colors.white_color,
            )
            self.transform = rendering.Transform()
            self.image_lock = threading.Lock()

        def make_circle(center: Tuple[int, int], radius: int, res=10) -> np.ndarray:
            """Create circle points

            Args:
                center (list): center of the circle
                radius (int): radius of the circle
                res (int, optional): resolution of points. Defaults to 10.

            Returns:
                list: vertices representing with desired resolution
            """
            thetas = np.linspace(0, 2 * np.pi, res + 1)[:-1]
            verts = np.zeros((res, 2))
            verts[:, 0] = center[0] + radius * np.cos(thetas)
            verts[:, 1] = center[1] + radius * np.sin(thetas)
            return verts

        with self.image_lock:
            self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()
            win.clear()
            gl.glViewport(0, 0, self.cfg.width, self.cfg.height)
            # Green background
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(*colors.bgcolor, 1.0)
            gl.glVertex3f(0, self.cfg.height, 0)
            gl.glVertex3f(self.cfg.width, self.cfg.height, 0)
            gl.glVertex3f(self.cfg.width, 0, 0)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            # Transform
            rob_x = self.robot.px
            rob_y = self.robot.py
            self.transform.enable()  # applies T_sim_in_viewport to below coords (all in sim frame)
            # Map closed obstacles ---
            self.obstacle_vertices = self.opt.contours
            for poly in self.obstacle_vertices:
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(*colors.obstcolor, 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
            # Agent body
            for idx, agent in enumerate([self.robot]):
                agent_x = agent.px
                agent_y = agent.py
                angle = self.robot.fix(agent.theta + np.pi / 2, 2 * np.pi)
                agent_r = agent.radius
                # Agent as Circle
                poly = make_circle((agent_x, agent_y), agent_r)
                gl.glBegin(gl.GL_POLYGON)
                if idx == 0:
                    color = np.array([1.0, 1.0, 1.0])
                else:
                    color = colors.agentcolor
                gl.glColor4f(*color, 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Direction triangle
                xnose = agent_x + agent_r * np.cos(angle)
                ynose = agent_y + agent_r * np.sin(angle)
                xright = agent_x + 0.3 * agent_r * -np.sin(angle)
                yright = agent_y + 0.3 * agent_r * np.cos(angle)
                xleft = agent_x - 0.3 * agent_r * -np.sin(angle)
                yleft = agent_y - 0.3 * agent_r * np.cos(angle)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(*colors.nosecolor, 1)
                gl.glVertex3f(xnose, ynose, 0)
                gl.glVertex3f(xright, yright, 0)
                gl.glVertex3f(xleft, yleft, 0)
                gl.glEnd()
            # Goal
            xgoal = self.robot.gx
            ygoal = self.robot.gy
            rgoal = self.robot.goal_radius
            # Goal markers
            gl.glBegin(gl.GL_POLYGON)
            gl.glColor4f(*colors.goalcolor, 1)
            triangle = make_circle((xgoal, ygoal), rgoal)
            for vert in triangle:
                gl.glVertex3f(vert[0], vert[1], 0)
            gl.glEnd()
            # Goal line
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glColor4f(*colors.goallinecolor, 1)
            gl.glVertex3f(rob_x, rob_y, 0)
            gl.glVertex3f(xgoal, ygoal, 0)
            gl.glEnd()
            # --
            self.transform.disable()
            self.iteration_label.text = f"Iter {self.opt.episode_steps}"
            self.iteration_label.draw()
            win.flip()
            if save_to_file:
                save_folder = path.join(
                    self.cfg.env_render_path, str(self.opt.episodes)
                )
                if not path.isdir(save_folder):
                    mkdir(save_folder)
                save_path = path.join(save_folder, f"{self.opt.episode_steps:08}.png")
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

    def _to_actionxy_format(self, action: np.ndarray) -> ActionXY:
        """Converts action array into action `ActionXY` object"""
        return ActionXY(action[0], action[1], 0)

    def reset(self) -> dict:
        """Resets robot state and generate new obstacles points

        Returns:
            dict: observation of the current environment state
        """
        if self.done or self.opt.is_initial_state:
            print("reseting robot env ...")
            self.robot.set_position(self.opt.robot_init_pos)
            self.robot.set_goal_position(self.opt.goal_init_pos)
            self.opt.total_reward += self.opt.episode_reward
            if self.opt.is_initial_state:
                self.results = []
                self.opt.total_reward = 0
                self.opt.total_steps = 0

            self.opt.success_flag = False
            self.opt.is_initial_state = False
            self.opt.episode_steps = 0
            self.done = False
            self.opt.episode_reward = 0
            (
                self.opt.flat_contours,
                self.opt.contours,
            ) = self.obstacles.get_flatten_contours()
        return self._make_obs()
