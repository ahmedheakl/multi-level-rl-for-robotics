
import pyglet
from pyglet.gl import *
from obstacle.obstacles import Obstacles
from obstacle.singleobstacle import SingleObstacle
from utils.action import ActionXY
from gym import Env


GREEN_COLOR = (50, 225, 30)
BLUE_COLOR = (0, 0, 255)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
MIN_SCREEN_WIDTH = 200
MIN_SCREEN_HEIGHT = 200


class Robot(object):
    def __init__(self, px=0, py=0, radius=0, vx=0, vy=0) -> None:
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
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


class Window(pyglet.window.Window, Env):
    def __init__(self, obstacles=None, *args, **kwargs):
        pyglet.window.Window.__init__(self, *args, **kwargs)
        self.set_minimum_size(MIN_SCREEN_WIDTH, MIN_SCREEN_HEIGHT)
        glClearColor(1, 1, 1, 1)
        self.robot = self.robot = Robot(px=500, py=100, radius=20, vx=0, vy=0)
        self.obstacles = obstacles
        self.obstacles_batch = pyglet.graphics.Batch()
        self.occupied = []
        for i in range(SCREEN_WIDTH):
            temp = []
            for j in range(SCREEN_HEIGHT):
                temp.append(False)
            self.occupied.append(temp)
        self.handle_obstacles()
        self.delta_t = 0.2

    def handle_obstacles(self):
        self.obsi = []
        for obstacle in self.obstacles.obstacles_list:
            self.obsi.append(pyglet.shapes.Rectangle(
                obstacle.px, obstacle.py, obstacle.width, obstacle.height, color=BLUE_COLOR, batch=self.obstacles_batch))
            for x in range(obstacle.px, obstacle.px+obstacle.width):
                for y in range(obstacle.py, obstacle.py+obstacle.height):
                    self.occupied[x][y] = True

    def valid(self, px, py):
        return px >= 0 and py >= 0 and py < SCREEN_HEIGHT and px < SCREEN_WIDTH

    def on_key_press(self, key, modifiers):
        if (key == pyglet.window.key.UP):
            action = ActionXY(0, 10)
            self.robot.step(action, self.delta_t)

        elif (key == pyglet.window.key.DOWN):
            action = ActionXY(0, -10)
            self.robot.step(action, self.delta_t)

        elif (key == pyglet.window.key.RIGHT):
            action = ActionXY(10, 0)
            self.robot.step(action, self.delta_t)

        elif (key == pyglet.window.key.LEFT):
            action = ActionXY(-10, 0)
            self.robot.step(action, self.delta_t)

    def on_draw(self):
        self.clear()
        self.robot_shape = pyglet.shapes.Circle(
            self.robot.px, self.robot.py, radius=self.robot.radius, color=GREEN_COLOR, batch=self.obstacles_batch)
        self.robot_shape.draw()
        self.obstacles_batch.draw()


if __name__ == '__main__':
    obs_lst = Obstacles([SingleObstacle(0, 0, 100, 100), SingleObstacle(
        150, 100, 300, 300)])
    window = Window(obstacles=obs_lst, width=SCREEN_WIDTH,
                    height=SCREEN_HEIGHT, resizable=True)
    pyglet.app.run()
