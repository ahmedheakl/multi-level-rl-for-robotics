from highrl.utils.abstract import Position

p1 = Position(1, 2)
p2 = Position(1, 3)
p3 = Position(2, 1)

ls = [p2, p1, p3]
print(sorted(ls))
