from highrl.obstacle.obstacles import Obstacles


class PlannerChecker:
    def __init__(self, height=0, width=0):
        self.dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        self.map = []
        self.height = height
        self.width = width
        self.visited = []
        self.DIRECTIONS = 8
        self.difficulity = 0

    def _check_valid_point(self, px, py):
        return px < self.height and py < self.width and px >= 0 and py >= 0

    def _reset(self, value):
        ret = []
        for x in range(self.height):
            current = []
            for y in range(self.width):
                current.append(value)
            ret.append(current)
        return ret

    def _construct_map(self, obstacles, new_obstacle=None):
        self.map = self._reset(".")
        self.visited = self._reset(False)
        for current_obstacle in obstacles.obstacles_list:
            cx, cy = map(int, [current_obstacle.px, current_obstacle.py])
            height, width = map(int, [current_obstacle.height, current_obstacle.width])
            for x in range(cx, min(cx + height, self.height)):
                for y in range(cy, min(cy + width, self.width)):
                    self.map[x][y] = "X"
        if new_obstacle != None:
            self.map[new_obstacle.px][new_obstacle.py] = "X"

    def _bfs(self, sx, sy, gx, gy):
        queue = []
        queue.append((sx, sy, 0))
        while len(queue) > 0:
            p = queue.pop(0)
            self.visited[p[0]][p[1]] = True
            if p[0] == gx and p[1] == gy:
                return p[2]
            for dir in range(self.DIRECTIONS):
                nx = p[0] + self.dx[dir]
                ny = p[1] + self.dy[dir]
                if self._check_valid_point(nx, ny) and self.map[nx][ny] == ".":
                    if self.visited[nx][ny] == True:
                        continue
                    self.visited[nx][ny] = True
                    queue.append((nx, ny, p[2] + 1))
        return 2000

    def get_map_difficulity(
        self,
        obstacles: Obstacles,
        height: int,
        width: int,
        sx: int,
        sy: int,
        gx: int,
        gy: int,
    ) -> int:
        self.height = height
        self.width = width
        self._construct_map(obstacles=obstacles)
        self.difficulity = self._bfs(sx, sy, gx, gy)
        self.visited = []
        self.map = []
        return self.difficulity

from typing import Tuple, List, Union
def get_region_coordinates(harmonic_number: int, eps: float, coords: List[Union[int, float]]) -> List[float]:
    px, py, gx, gy = coords
    slope = (gy - py) / (gx - px)
    intercept = (gx*py - gy*px) / (gx-py)
    shift_amount = eps * harmonic_number
    left_line = lambda x: [slope * x + (intercept + shift_amount), x]
    right_line = lambda x: [slope * x + (intercept - shift_amount), x]
    """
    p1-----p2
    |       |
    |       |
    p3-----p4
    """
    x1: float = (1 / slope) * ((gx*py - gy*gx) / (px - gx) - shift_amount / 2)   
    x2: float = (1 / slope) * ((gx*py - gy*gx) / (px - gx) + shift_amount / 2)  
    x3: float = (1 / slope) * ((gy*px - px*py) / (gx - px) - shift_amount / 2)
    x4: float = (1 / slope) * ((gy*px - px*py) / (gx - px) - shift_amount / 2)     
    x_coords = [x1, x2, x3, x4]
    points = [right_line(x_coords[i]) if i&1 else left_line(x_coords[i]) for i in range(4)]
    return points
    
    
    
    
def convex_hull_difficulty(obstacles: Obstacles) -> float:
    pass