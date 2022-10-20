from obstacle.singleobstacle import SingleObstacle
import abc
import random


class Obstacles(object):
    def __init__(self, obstacles_list=[]) -> None:
        self.obstacles_list = obstacles_list

    @abc.abstractmethod
    def generate_random_obstacles(self, number_of_obstacles=5, max_width=20, max_height=20, map_width=100, map_height=100):
        current_obstacles = set()
        for obstacle in range(number_of_obstacles):
            while (len(current_obstacles) < obstacle+1):
                cur_width = random.choice(1, max_width)
                cur_height = random.choice(1, max_height)
                cur_px = random.choice(map_height - max_height)
                cur_py = random.choice(map_width - max_width)
                current_obstacles.add([cur_px, cur_py, cur_width, cur_height])
        while (len(current_obstacles) > 0):
            px, py, width, height = current_obstacles.pop()
            obstacle_obj = SingleObstacle(
                px=px, py=py, width=width, height=height)
            self.obstacles_list.append(obstacle_obj)
