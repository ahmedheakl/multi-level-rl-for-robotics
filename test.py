from obstacle.obstacles import Obstacles
from obstacle.singleobstacle import SingleObstacle
from utils.planner_checker import PlannerChecker

plannerChecker = PlannerChecker(1280, 720)  # camel Case 


obs_lst = Obstacles([SingleObstacle(0, 0, 100, 100), SingleObstacle(
            350, 400, 300, 300)])


print(plannerChecker.get_map_difficulity(obs_lst, 400, 400, 100, 150, 100, 400))
