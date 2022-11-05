from cmath import pi
from math import cos, sin
import numpy as np


# the lidar 1D reading must be an perfect square number
# the 1D lidar array should be arranged startig from the distance towards the positive x-axis
#
class LidarRings:
    def __init__(self, lidar_1D, obs_size = 64, original_env_size = 720, px=0, py=0):
        self.conversion_factor = obs_size / original_env_size
        self.lidar_1D = lidar_1D
        self.obs_size = obs_size
        self.x = np.zeros(len(self.lidar_1D), dtype=np.int64)
        self.y = np.zeros(len(self.lidar_1D), dtype=np.int64)
        self.px = px
        self.py = py
        self.pi = 22 / 7
        self.angle = np.zeros(len(self.lidar_1D), dtype=np.float64)
        self.angle_cos = np.zeros(len(self.lidar_1D), dtype=np.float64)
        self.angle_sin = np.zeros(len(self.lidar_1D), dtype=np.float64)
        for i in range(len(self.lidar_1D)):
            self.angle[i] = ((360 / len(self.lidar_1D)) * i) * (self.pi / 180)
            self.angle_cos[i] = np.rint(cos(self.angle[i]))
            self.angle_sin[i] = np.rint(sin(self.angle[i]))
        return

    def extract_x_and_y(self):
        for i in range(len(self.lidar_1D)):
            # print("iter {}".format(i))
            self.x[i] = np.clip(
                (self.lidar_1D[i] * self.angle_cos[i]), a_min=0, a_max=720
            )
            self.y[i] = np.clip(
                (self.lidar_1D[i] * self.angle_sin[i]), a_min=0, a_max=720
            )
            # print("angle {}".format(self.angle[i]*180/self.pi))
            # print("cos  = {}, sin = {}".format(self.angle_cos[i], self.angle_sin[i]))
            # print("x = {}, y = {}".format(self.x,self.y))
            self.x[i] = np.rint((self.x[i] + self.px) * self.conversion_factor)
            self.y[i] = np.rint((self.y[i] + self.py) * self.conversion_factor)

            # if (self.x[i] > self.env_side):
            #     self.x[i] = 0
            # if (self.y[i] > self.env_side):
            #     self.y[i] = 0
        output = (self.x, self.y)
        return output

    def generate_2d_lidar_obs(self, x, y):
        self.x = x
        self.y = y
        # self.env_side = int(self.env_size**0.5)
        self.pic = np.random.randint(
            1, size=(self.obs_size, self.obs_size), dtype=np.uint8
        )

        # for i in range(self.env_side):
        #     for j in range(self.env_side):
        #         for z in range(len(self.x)):
        #             if (i == self.x[z]) and (j == self.y[z]):
        #                 self.pic[self.y[z]][self.x[z]] = 1
        for i in range(len(self.x)):
            if self.x[i] >= 0 and self.y[i] >= 0:
                if self.x[i] <= self.obs_size and self.y[i] <= self.obs_size:
                    self.pic[self.y[i] - 1][self.x[i] - 1] = 1
        # print(self.x, self.y)
        copy = self.pic.copy()

        for i in range(self.obs_size):
            self.pic[i] = copy[self.obs_size - 1 - i]
        self.pic = 1 - self.pic
        return self.pic


if __name__ == "__main__":
    lidar_rings = LidarRings(lidar_1D = [1, 1, 1, 1], obs_size = 64, original_env_size = 720, px=0, py=0)
    x, y = lidar_rings.extract_x_and_y()
    obs = lidar_rings.generate_2d_lidar_obs(x, y)
    print(obs)
