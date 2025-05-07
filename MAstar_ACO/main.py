"""
@author: Dz-ZiYang Deng
"""
import matplotlib
import planner_config as cfg  # User configurations
from planner_core import PathPlanner  # The main planner class

if __name__ == "__main__":
    try:
        matplotlib.rcParams['font.family'] = cfg.FONT_FAMILY
        matplotlib.rcParams['font.sans-serif'] = [cfg.FONT_FAMILY]
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = cfg.FONT_SIZE
    except Exception as e:
        print(f"Note: Could not set Matplotlib font to '{cfg.FONT_FAMILY}'. System may not have it. Error: {e}")
        print("Chinese characters might not display correctly.")

    # Initialize the PathPlanner with parameters from the config file or defaults
    planner_app = PathPlanner(
        width=cfg.GRID_WIDTH,
        height=cfg.GRID_HEIGHT,
        grid_size=cfg.GRID_CELL_SIZE,
        obstacle_prob=cfg.OBSTACLE_PROBABILITY
    )

    planner_app.run()