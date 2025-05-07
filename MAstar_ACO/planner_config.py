"""
@author: Dz-ZiYang Deng
"""
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_CELL_SIZE = 20
OBSTACLE_PROBABILITY = 0.25 # Default obstacle probability

# Matplotlib Font Settings (can be set in main.py as well)
FONT_FAMILY = 'SimHei' # Or 'Microsoft YaHei', 'Arial Unicode MS' etc. for Chinese
FONT_SIZE = 10

# MultiA* / A* Parameters

# Ant Colony Optimization (ACO) Parameters
ACO_NUM_ANTS = 70               # 增加蚂蚁数量
ACO_ITERATIONS = 30             # 增加迭代次数
ACO_ALPHA = 1.0                 # Pheromone importance factor
ACO_BETA = 4.0                  # 增加启发式因子 (从2.0增加到2.5)
ACO_RHO = 0.35                   # 增加信息素蒸发率 (从0.1增加到0.2)
ACO_Q = 100                     # Pheromone deposit strength factor
ACO_INITIAL_PHEROMONE = 0.1
ACO_PHEROMONE_MIN = 0.01
ACO_PHEROMONE_MAX = 10.0

# Path Smoothing Parameters
SMOOTHING_ITERATIONS = 70       # 增加平滑迭代次数 (从50增加到70)
SMOOTHING_ALPHA = 0.2           # 增加权重 (从0.2增加到0.3)
SMOOTHING_BETA = 0.18            # 增加权重 (从0.2增加到0.3)

