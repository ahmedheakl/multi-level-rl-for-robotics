import pyglet
from pyglet.gl import *
import numpy as np

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


class Window(pyglet.window.Window):
    def __init__(self, obstacles=None, robot=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(MIN_SCREEN_WIDTH, MIN_SCREEN_HEIGHT)
        glClearColor(1, 1, 1, 1)
        self.obstacles = obstacles
        self.obstacles_batch = pyglet.graphics.Batch()
        self.lines_batch = pyglet.graphics.Batch()
        self.robot = robot
        self.handle_obstacles()

    def handle_obstacles(self):
        self.obsi = []
        for obstacle in self.obstacles.obstacles_list:
            self.obsi.append(pyglet.shapes.Rectangle(
                obstacle.px, obstacle.py, obstacle.width, obstacle.height, color=BLUE_COLOR, batch=self.obstacles_batch))

    def handle_lines(self, ranges=[], angles=[]):
        self.ranges = ranges
        self.angles = angles
        self.lines = []
        for i in range(len(self.ranges)):
            x1, y1 = self.robot.px, self.robot.py
            x2, y2 = self.robot.px + self.ranges[i] * np.cos(
                self.angles[i]), self.robot.py + self.ranges[i] * np.sin(self.angles[i])
            x1, y1 = clip(x1, 0, SCREEN_WIDTH), clip(y1, 0, SCREEN_HEIGHT)
            x2, y2 = clip(x2, 0, SCREEN_WIDTH), clip(y2, 0, SCREEN_HEIGHT)
            print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            line = pyglet.shapes.Line(
                x1, y1, x2, y2, 10, color=BLACK_COLOR)
            line.draw()

    def valid(self, px, py):
        return px >= 0 and py >= 0 and py < SCREEN_HEIGHT and px < SCREEN_WIDTH

    def on_draw(self):
        self.clear()
        # self.handle_lines(ranges=ranges, angles=angles)
        self.robot_shape = pyglet.shapes.Circle(
            self.robot.px, self.robot.py, radius=self.robot.radius, color=GREEN_COLOR, batch=self.obstacles_batch)
        self.robot_shape.draw()
        self.obstacles_batch.draw()
