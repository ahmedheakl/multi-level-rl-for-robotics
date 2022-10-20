import abc


class SingleObstacle(object):
    def __init__(self, px=None, py=None, width=None, height=None):
        self.px = px
        self.py = py
        self.width = width
        self.height = height

    def get_position(self):
        return self.px, self.py

    def get_points(self):
        """
        p1----p2
        |     |
        |     |
        p4----p3
        """
        return [(self.px, self.py), (self.px + self.width, self.py), (
                self.px+self.width, self.py + self.height), (self.px, self.py + self.height)]

    def get_dimension(self):
        return self.width, self.height
