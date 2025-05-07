"""
@author: Dz-ZiYang Deng
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from tkinter import messagebox, Tk  # For error/info dialogs
import math
import random
import time  # For measuring performance

# Import your optimized algorithms and config
import algorithms
import planner_config as cfg


class PathPlanner:
    def __init__(self, width, height, grid_size, obstacle_prob):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = width // grid_size
        self.rows = height // grid_size
        self.obstacle_prob = obstacle_prob

        self.start_node = None
        self.goal_node = None
        self.waypoints = []

        self.path_mmstar = []
        self.path_aco_smoothed = []
        self.total_planning_time = 0.0
        self.robot_state_world = None

        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.generate_obstacles()

        self.config_params = {name: getattr(cfg, name) for name in dir(cfg) if not name.startswith('__')}

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('MultiA* - ACO Path Planner')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_event)

        self.add_buttons()
        self.update_plot()

    def generate_obstacles(self):
        self.grid.fill(0)
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < self.obstacle_prob:
                    self.grid[r, c] = 1

        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        if self.start_node and self.grid[self.start_node[0], self.start_node[1]] == 1:
            self.grid[self.start_node[0], self.start_node[1]] = 0
        if self.goal_node and self.grid[self.goal_node[0], self.goal_node[1]] == 1:
            self.grid[self.goal_node[0], self.goal_node[1]] = 0
        for wp in self.waypoints:
            if self.grid[wp[0], wp[1]] == 1:
                self.grid[wp[0], wp[1]] = 0

    def add_buttons(self):
        """Add control buttons to the UI"""
        ax_clear = plt.axes([0.81, 0.02, 0.09, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_all_points_paths)

        ax_new_map = plt.axes([0.71, 0.02, 0.09, 0.04])
        self.btn_new_map = Button(ax_new_map, 'New Map')
        self.btn_new_map.on_clicked(self.reset_map_and_clear_all)

        ax_plan = plt.axes([0.61, 0.02, 0.09, 0.04])
        self.btn_plan = Button(ax_plan, 'Plan Path')
        self.btn_plan.on_clicked(lambda event: self.initiate_path_planning())

    def clear_all_points_paths(self, event=None):
        self.start_node = None
        self.goal_node = None
        self.waypoints = []
        self.path_mmstar = []
        self.path_aco_smoothed = []
        self.robot_state_world = None
        print("All points and paths cleared.")
        self.update_plot()

    def reset_map_and_clear_all(self, event=None):
        self.clear_all_points_paths()
        self.generate_obstacles()
        print("New map generated and all cleared.")
        self.update_plot()

    def on_click_event(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata // self.grid_size)
        row = int(event.ydata // self.grid_size)

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if self.grid[row, col] == 1:
            print("Clicked on an obstacle.")
            return

        clicked_grid_node = (row, col)

        if event.button == 1:
            if self.start_node == clicked_grid_node:
                self.start_node = None
                print(f"Start node at {clicked_grid_node} removed.")
            else:
                if self.start_node:
                    print(f"Previous start node at {self.start_node} replaced by {clicked_grid_node}.")
                else:
                    print(f"Start node set to: {clicked_grid_node}")
                self.start_node = clicked_grid_node

            if self.goal_node == self.start_node:
                self.goal_node = None
                print("Goal node was same as start node, goal node cleared.")

            if self.start_node in self.waypoints:
                self.waypoints.remove(self.start_node)
                print(f"Waypoint at {self.start_node} removed as it became the start node.")

        elif event.button == 3:
            if clicked_grid_node == self.start_node or clicked_grid_node == self.goal_node:
                print("Cannot set a waypoint on the start or goal node.")
            elif clicked_grid_node in self.waypoints:
                self.waypoints.remove(clicked_grid_node)
                print(f"Waypoint at {clicked_grid_node} removed.")
            else:
                self.waypoints.append(clicked_grid_node)
                print(f"Waypoint added at: {clicked_grid_node}")

        elif event.button == 2:
            if self.goal_node == clicked_grid_node:
                self.goal_node = None
                print(f"Goal node at {clicked_grid_node} removed.")
            else:
                if self.goal_node:
                    print(f"Previous goal node at {self.goal_node} replaced by {clicked_grid_node}.")
                else:
                    print(f"Goal node set to: {clicked_grid_node}")
                self.goal_node = clicked_grid_node

            if self.start_node == self.goal_node:
                self.start_node = None
                print("Start node was same as goal node, start node cleared.")

            if self.goal_node in self.waypoints:
                self.waypoints.remove(self.goal_node)
                print(f"Waypoint at {self.goal_node} removed as it became the goal node.")

        self.path_mmstar = []
        self.path_aco_smoothed = []
        self.robot_state_world = None

        self.update_plot()

    def initiate_path_planning(self, event=None):
        if not self.start_node or not self.goal_node:
            messagebox.showerror("Planning Error", "Start and Goal nodes must be set before planning.")
            print("Error: Start and/or Goal node not set.")
            return

        print(f"\n--- Initiating Path Planning ---")
        print(f"From: {self.start_node} To: {self.goal_node} Via: {self.waypoints}")


        self.path_mmstar = []
        self.path_aco_smoothed = []
        self.robot_state_world = None
        self.update_plot()
        plt.pause(0.01)

        start_time = time.time()

        print("1. Running Path Search...")
        t0 = time.time()
        try:
            if hasattr(algorithms, 'mm_star_search'):
                self.path_mmstar = algorithms.mm_star_search(
                    self.start_node, self.goal_node, self.waypoints,
                    self.rows, self.cols, self.grid, self.config_params
                )
            # Fallback to a_star if available
            elif hasattr(algorithms, 'a_star_search'):
                print("MultiA* function not found, falling back to A* search")
                if not self.waypoints:
                    self.path_mmstar = algorithms.a_star_search(
                        self.start_node, self.goal_node,
                        self.rows, self.cols, self.grid, self.config_params
                    )
                else:
                    print(f"Planning path through {len(self.waypoints)} waypoints")
                    self.path_mmstar = []
                    current = self.start_node
                    all_points = self.waypoints.copy() + [self.goal_node]

                    for next_point in all_points:
                        segment = algorithms.a_star_search(
                            current, next_point,
                            self.rows, self.cols, self.grid, self.config_params
                        )
                        if not segment:
                            print(f"Failed to find path to waypoint {next_point}")
                            break
                        if self.path_mmstar:
                            self.path_mmstar.extend(segment[1:])
                        else:
                            self.path_mmstar.extend(segment)
                        current = next_point
            else:
                raise AttributeError("No path search function found in algorithms module")
        except Exception as e:
            print(f"Path search error: {e}")
            messagebox.showerror("Path Planning Error", f"Failed to find path: {str(e)}")
            return
        t1 = time.time()
        if not self.path_mmstar:
            messagebox.showerror("MultiA* Error", "MultiA* algorithm failed to find a path.")
            print("MultiA* Error: Path not found.")
            self.update_plot()
            return
        print(f"MultiA* Path Found: {len(self.path_mmstar)} nodes in {t1 - t0:.3f} seconds.")
        self.update_plot()
        plt.pause(0.01)

        print("2. Running Ant Colony Optimization...")
        t0 = time.time()
        try:
            if hasattr(algorithms, 'ant_colony_optimization'):
                self.path_aco_smoothed = algorithms.smooth_path(
                    self.path_mmstar, self.grid, self.rows, self.cols, self.config_params
                )
            elif hasattr(algorithms, 'aco_path_planning') or hasattr(algorithms, 'aco'):
                aco_func = getattr(algorithms, 'aco_path_planning', None) or getattr(algorithms, 'aco')
                self.path_aco_smoothed = aco_func(
                    self.path_mmstar, self.rows, self.cols, self.grid, self.config_params
                )
            elif hasattr(algorithms, 'smooth_path'):
                print("ACO function not found, falling back to path smoothing")
                self.path_aco_smoothed = algorithms.smooth_path(
                    self.path_mmstar, self.grid, self.rows, self.cols, self.config_params
                )
            else:
                print("Warning: No ACO or smoothing function found, using raw path")
                self.path_aco_smoothed = list(self.path_mmstar)
        except Exception as e:
            print(f"ACO optimization error: {e}")
            print("Using MultiA* path as fallback.")
            self.path_aco_smoothed = list(self.path_mmstar)

        t1 = time.time()
        if not self.path_aco_smoothed:
            print("ACO Error: Optimization failed. Using MultiA* path.")
            self.path_aco_smoothed = list(self.path_mmstar)
        else:
            print(f"ACO Path Found: {len(self.path_aco_smoothed)} nodes in {t1 - t0:.3f} seconds.")

        # Total execution time
        total_time = time.time() - start_time
        self.total_planning_time = total_time
        print(f"Total planning time: {total_time:.3f} seconds")

        print("--- Path Planning Complete ---")
        self.update_plot()
        if self.path_mmstar and self.path_aco_smoothed:
            mmstar_len = self.compute_path_distance(self.path_mmstar)
            aco_len = self.compute_path_distance(self.path_aco_smoothed)

            print("\n=== 路径长度对比 ===")
            print(f"MultiA* 路径总长度: {mmstar_len:.2f}")
            print(f"ACO 优化后路径总长度: {aco_len:.2f}")
            print(f"优化减少: {mmstar_len - aco_len:.2f} 单位")

        print(f"[DEBUG] MultiA* Path Length: {len(self.path_mmstar)}")
        print(f"[DEBUG] MultiA* Path: {self.path_mmstar}")

    def update_plot(self):
        self.ax.clear()

        # Draw grid and obstacles
        for r_idx in range(self.rows):
            for c_idx in range(self.cols):
                color = 'black' if self.grid[r_idx, c_idx] == 1 else 'white'
                edge_color = 'gray' if self.grid[r_idx, c_idx] == 0 else 'black'
                rect = patches.Rectangle(
                    (c_idx * self.grid_size, r_idx * self.grid_size),
                    self.grid_size, self.grid_size,
                    linewidth=0.5, edgecolor=edge_color, facecolor=color,
                    alpha=0.7 if color == 'white' else 1
                )
                self.ax.add_patch(rect)

        # Draw Start, Goal, Waypoints (as grid nodes)
        if self.start_node:
            r, c = self.start_node
            self.ax.plot(
                c * self.grid_size + self.grid_size / 2,
                r * self.grid_size + self.grid_size / 2,
                'go', markersize=10, label='Start'
            )

        if self.goal_node:
            r, c = self.goal_node
            self.ax.plot(
                c * self.grid_size + self.grid_size / 2,
                r * self.grid_size + self.grid_size / 2,
                'ro', markersize=10, label='Goal'
            )

        for i, (r, c) in enumerate(self.waypoints):
            self.ax.plot(
                c * self.grid_size + self.grid_size / 2,
                r * self.grid_size + self.grid_size / 2,
                'bo', markersize=7, label='Waypoint' if i == 0 else ""
            )

        # Draw paths (list of grid nodes)
        path_params = {'linewidth': 2, 'alpha': 0.8}

        if self.path_mmstar:
            path_x = [(n[1] + 0.5) * self.grid_size for n in self.path_mmstar]
            path_y = [(n[0] + 0.5) * self.grid_size for n in self.path_mmstar]
            self.ax.plot(path_x, path_y, 'm-', **path_params, label='MultiA* Path')

        if self.path_aco_smoothed:
            path_x = [(n[1] + 0.5) * self.grid_size for n in self.path_aco_smoothed]
            path_y = [(n[0] + 0.5) * self.grid_size for n in self.path_aco_smoothed]
            self.ax.plot(path_x, path_y, 'c--', linewidth=2, alpha=0.5, label='ACO/Smoothed Path')

        # Set plot limits and labels
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal', adjustable='box')
        title_str = "MultiA*-ACO Path Planner (左键:Start, 右键:Waypoint, 中键:Goal)"
        self.ax.set_title(title_str, fontdict={'fontname': cfg.FONT_FAMILY, 'fontsize': cfg.FONT_SIZE + 2})

        # Add legend
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            self.ax.legend(
                by_label.values(), by_label.keys(),
                loc='upper left',
                prop={'size': 8, 'family': cfg.FONT_FAMILY}
            )
        xtick_positions = np.arange(0, self.cols + 1, 5) * self.grid_size
        ytick_positions = np.arange(0, self.rows + 1, 5) * self.grid_size
        xtick_labels = [str(i) for i in range(0, self.cols + 1, 5)]
        ytick_labels = [str(i) for i in range(0, self.rows + 1, 5)]

        self.ax.set_xticks(xtick_positions)
        self.ax.set_xticklabels(xtick_labels)

        self.ax.set_yticks(ytick_positions)
        self.ax.set_yticklabels(ytick_labels)

        self.ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
        if hasattr(self, 'info_text_obj'):
            self.info_text_obj.remove()

        if self.path_aco_smoothed:
            path_len = self.compute_path_distance(self.path_aco_smoothed)
            info_text = f"路径长度：{path_len:.2f}  |  总耗时：{self.total_planning_time * 1000:.1f} ms"

            self.info_text_obj = self.fig.text(
                0.05, 0.02,
                info_text,
                fontsize=11,
                fontname=cfg.FONT_FAMILY,
                ha='left',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
            )
        self.fig.canvas.draw_idle()



    def run(self):
        print("Planner running. Please interact with the map:")
        print("- Left-click: Set/Change Start Node. Click again on the start node to remove it.")
        print("- Right-click: Add Waypoint. Click again on a waypoint to remove it.")
        print("- Middle-click: Set/Change Goal Node. Click again on the goal node to remove it.")
        print("- Buttons: 'Clear All', 'New Map', 'Plan Path'")
        plt.show()

    def compute_path_distance(self, path):
        from algorithms import euclidean_distance
        return sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
