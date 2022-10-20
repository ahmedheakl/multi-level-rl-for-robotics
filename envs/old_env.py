from gym import Env, spaces
from utils.action import *
import random
from obstacle.obstacles import Obstacles
from obstacle.singleobstacle import SingleObstacle
import numpy as np
from numpy.linalg import norm
from CMap2D import flatten_contours, render_contours_in_lidar, CMap2D, CSimAgent, fast_2f_norm
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from utils.calculations import *
from utils.basepygletenv import Window
import pyglet


GREEN_COLOR = (50, 225, 30)
BLUE_COLOR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MIN_SCREEN_WIDTH = 200
MIN_SCREEN_HEIGHT = 200


def clip(val, min_val, max_val):
    if (val < min_val):
        return min_val
    if (val > max_val):
        return max_val
    return val


class Robot(object):
    def __init__(self, px=0, py=0, radius=0, vx=0, vy=0, gx=200, gy=200) -> None:
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.radius = radius

    def calc_pos(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    def step(self, action, delta_t):
        pos = self.calc_pos(action, delta_t)
        self.px, self.py = pos
        self.vx = action.vx
        self.vy = action.vy

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius


class TestEnv(Env):
    def __init__(self, width=1280, height=720, resizable=True) -> None:
        super(TestEnv, self).__init__()
        self.action_space_names = ["ActionXY", "ActionRot"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf,
                       shape=(1080,), dtype=np.float32),
            spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        ))
        obs_lst = Obstacles([SingleObstacle(0, 0, 100, 100), SingleObstacle(
            150, 100, 300, 300)])
        self.obstacles = obs_lst
        self.width = width
        self.height = height
        self.resizable = resizable
        self.delta_t = 0.2
        self.robot = Robot(px=500, py=100, radius=20, vx=0, vy=0)
        self.window = None
        self.n_angles = 1080
        self.lidarAngleIncrement = 0.00581718236208  # = 1/3 degrees
        self.lidarMinAngle = 0
        self.lidarMaxAngle = 6.27543783188 + self.lidarAngleIncrement  # = 2*pi
        self.lidarScan = None
        self.converterCMap2D = CMap2D()
        self.converterCMap2D.set_resolution(1.)

    def generate_obstacles_points(self):
        self.contours = []
        for obstacle in self.obstacles.obstacles_list:
            self.contours.append([(obstacle.px, obstacle.py), (obstacle.px + obstacle.width, obstacle.py), (
                obstacle.px, obstacle.py + obstacle.height), (obstacle.px + obstacle.width, obstacle.py + obstacle.height)])
        self.flat_contours = flatten_contours(self.contours)

    def _make_obs(self):
        robot = self.window.robot
        lidar_pos = np.array([robot.px, robot.py, 0], dtype=np.float32)
        ranges = np.ones((self.n_angles,), dtype=np.float32) * 200.
        angles = np.linspace(self.lidarMinAngle,
                             self.lidarMaxAngle-self.lidarAngleIncrement,
                             self.n_angles) + lidar_pos[2]
        self.generate_obstacles_points()
        render_contours_in_lidar(
            ranges, angles, self.flat_contours, lidar_pos[:2])
        self.lidar_scan = ranges
        self.lidar_angles = angles

        baselink_in_world = np.array([robot.px, robot.py, 0])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        # TODO: actual robot rot vel?
        robotvel_in_world = np.array([robot.vx, robot.vy, 0])
        robotvel_in_baselink = apply_tf_to_vel(
            robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack(
            [goal_in_baselink[:2], robotvel_in_baselink])
        obs = (self.lidar_scan, robotstate_obs)

        return obs

    def render(self):
        self.window.on_draw()
        return self.window

    def close(self):
        self.window.close()

    def detect_collison(self, robot):
        ok = True
        for obstacle in self.obstacles.obstacles_list:
            distances = point_to_obstacle_distance(robot, obstacle)
            for dist in distances:
                ok &= (dist > robot.radius)
        return ok

    def step(self, action):
        """
        To do:
        - move human
        - return state
        - Calculate reward
        """

        assert type(
            action).__name__ in self.action_space_names, "Invalid Action"

        robot = self.window.robot
        old_distance_to_goal = point_to_point_distance(
            (robot.px, robot.py), (robot.gx, robot.gy))
        robot.step(action, self.delta_t)

        reward = 0

        new_distance_to_goal = point_to_point_distance(
            (robot.px, self.robot.py), (robot.gx, robot.gy))

        reward += (old_distance_to_goal - new_distance_to_goal)

        done = robot.reached_destination()
        reward += done * 100

        if (self.detect_collison(self.window.robot)):
            reward = -100
            done = True

        self.window.robot = robot

        return self._make_obs(), reward, done

    def reset(self):
        # self.generate_obstacles_points()
        self.robot = Robot(px=100, py=150, radius=20)
        self.window = Window(obstacles=self.obstacles, robot=self.robot, width=self.width,
                             height=self.height, resizable=self.resizable)
        return self._make_obs()


if __name__ == '__main__':
    # obs_lst = Obstacles([SingleObstacle(0, 0, 100, 100), SingleObstacle(
    #     150, 100, 300, 300)])
    # env = TestEnv(obstacles=obs_lst, width=SCREEN_WIDTH,
    #               height=SCREEN_HEIGHT, resizable=True)
    env = TestEnv(width=SCREEN_WIDTH,
                  height=SCREEN_HEIGHT, resizable=True)
    res = env.reset()
    window = env.render()

    obs = None
    reward = 0

    def update(dt):
        global reward
        global obs
        global window
        vx_sample = random.randint(2, 20)
        vy_sample = random.randint(2, 20)
        action = ActionXY(vx_sample, vy_sample)
        obs, reward, done = env.step(action)
        print("Reward = {0:.3f}".format(reward), done)
        if (done):
            window.close()
        else:
            window = env.render()

    pyglet.clock.schedule_interval(update, 0.1)
    pyglet.app.run()
