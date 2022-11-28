from highrl.obstacle.single_obstacle import SingleObstacle
import numpy as np


class Obstacles(object):
    def __init__(self, obstacles_list=[]) -> None:
        self.obstacles_list = obstacles_list
        self.num = 0

    def __add__(self, obstacle: SingleObstacle):
        self.obstacles_list.append(obstacle)
        return self

    def get_flatten_contours(self):
        contours = []
        for obstacle in self.obstacles_list:
            contours.append(obstacle.get_points())
        contours = np.array(contours)
        n_total_vertices = int(
            np.sum([len(verts) for verts in contours]) + len(contours)
        )
        flat_contours = np.zeros((n_total_vertices, 3), dtype=np.float32)
        v = 0
        for idx, polygon in enumerate(contours):
            # add first vertex last to close polygon
            for vertex in np.concatenate((polygon, polygon[:1]), axis=0):
                flat_contours[v, :] = np.array([idx, vertex[0], vertex[1]])
                v += 1
        return flat_contours, contours

    def __iter__(self):
        return self

    def __next__(self):
        if self.num == len(self.obstacles_list):
            raise StopIteration
        self.num += 1
        return self.obstacles_list[self.num - 1]

    def __str__(self):
        ret = "[\n"
        for obstacle in self.obstacles_list:
            ret += f"{obstacle}\n"
        ret += "]"
        return ret
