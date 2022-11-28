from highrl.obstacle import Obstacles, SingleObstacle
from highrl.agents.robot import Robot
from highrl.utils.planner_checker import convex_hull_difficulty
obstacles = Obstacles([SingleObstacle(-1, 0, 1, 720), SingleObstacle(0, -1, 1280, 1), SingleObstacle(1280, 0, 1, 720), SingleObstacle(0, 720, 1280, 1), SingleObstacle(514, 423, 43, 12)])
robot = Robot()
robot.set(px=1151, py=107, gx=219, gy=7)
width = 1280
height = 720
print(convex_hull_difficulty(obstacles, robot, width, height))