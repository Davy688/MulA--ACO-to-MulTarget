# visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib  # For font setting
from matplotlib.path import Path
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines

# Setup Chinese font support (optional, keep if needed)
try:
    matplotlib.rcParams['font.family'] = 'SimHei'  # Or another installed Chinese font
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Note: Could not set Chinese font. Error: {e}")


def create_star(center, size=0.35):
    """
    Creates a path for a 5-pointed star.

    Args:
        center (tuple): Center point (x, y) for the star
        size (float): Size of the star

    Returns:
        matplotlib.path.Path: Path for the star shape
    """
    x, y = center
    # Create star shape vertices (5 outer points, 5 inner points)
    outer_radius = size
    inner_radius = size * 0.4
    angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, 10, endpoint=False)

    # Alternate between outer and inner radius points
    radii = np.array([outer_radius, inner_radius] * 5)
    points = np.zeros((10, 2))
    points[:, 0] = x + radii * np.cos(angles)
    points[:, 1] = y + radii * np.sin(angles)

    # Define the path codes
    codes = [Path.MOVETO] + [Path.LINETO] * 9 + [Path.CLOSEPOLY]
    vertices = np.vstack([points, points[0]])  # Close the path

    return Path(vertices, codes)


def visualize_map_only(grid, obstacle_set, map_size, start, goal, output_filename=None):
    """
    Visualizes only the map grid with start and goal points, without any paths.

    Args:
        grid (list[list[int]]): The map grid.
        obstacle_set (set[tuple[int, int]]): Set of obstacle coordinates (x, y).
        map_size (int): Dimension of the map.
        start (tuple[int, int]): Start coordinate (x, y).
        goal (tuple[int, int]): Goal coordinate (x, y).
        output_filename (str, optional): If provided, saves the figure to this file.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw Grid and Obstacles
    for r in range(map_size):
        for c in range(map_size):
            is_obstacle = (c, r) in obstacle_set
            color = '#444444' if is_obstacle else 'white'
            rect = patches.Rectangle((c, r), 1, 1, linewidth=0.5,
                                     edgecolor='#DDDDDD', facecolor=color)
            ax.add_patch(rect)

    # Mark Start as a circle
    start_x, start_y = start[0] + 0.5, start[1] + 0.5
    start_patch = patches.Circle((start_x, start_y), 0.3,
                                 facecolor='#33FF33', edgecolor='black', label='Start', zorder=10)
    ax.add_patch(start_patch)

    # Mark Goal as a star
    goal_x, goal_y = goal[0] + 0.5, goal[1] + 0.5
    star_path = create_star((goal_x, goal_y))
    goal_patch = patches.PathPatch(star_path, facecolor='#FF3333',
                                   edgecolor='black', label='Goal', zorder=10)
    ax.add_patch(goal_patch)

    # Configure Axes
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, map_size + 1, 5))
    ax.set_yticks(np.arange(0, map_size + 1, 5))
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(False)

    # Add Title
    plt.title('路径规划地图', pad=15, fontsize=14)

    # Create legend elements with the actual shapes
    # For start point (green circle)
    start_legend = mlines.Line2D([], [], marker='o', markerfacecolor='#33FF33',
                                 markeredgecolor='black', markersize=10, linestyle='None',
                                 label='起点')

    # For goal point (red star)
    goal_legend = mlines.Line2D([], [], marker='*', markerfacecolor='#FF3333',
                                markeredgecolor='black', markersize=10, linestyle='None',
                                label='终点')

    # For obstacles (black square)
    obstacle_legend = patches.Patch(facecolor='#444444', edgecolor='#DDDDDD', label='障碍物')

    # Combine legend elements
    legend_elements = [start_legend, goal_legend, obstacle_legend]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save if filename provided
    if output_filename:
        try:
            plt.savefig(output_filename, dpi=200, bbox_inches='tight')
            print(f"Map visualization saved to {output_filename}")
        except Exception as e:
            print(f"Error saving map visualization: {e}")

    plt.show()


def visualize_paths_single_map(grid, obstacle_set, paths_dict, run_times, map_size, start, goal):
    """
    Visualizes multiple paths on a single map grid.

    Args:
        grid (list[list[int]]): The map grid (not directly used for plotting obstacles here).
        obstacle_set (set[tuple[int, int]]): Set of obstacle coordinates (x, y).
        paths_dict (dict): Dictionary {AlgorithmName: path_list}.
                           path_list is a list of (x, y) tuples.
        run_times (dict): Dictionary {AlgorithmName: time_float}.
        map_size (int): Dimension of the map.
        start (tuple[int, int]): Start coordinate (x, y).
        goal (tuple[int, int]): Goal coordinate (x, y).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Draw Grid and Obstacles ---
    for r in range(map_size):
        for c in range(map_size):
            is_obstacle = (c, r) in obstacle_set
            color = '#444444' if is_obstacle else 'white'  # Darker obstacles
            # Draw rectangle for the cell
            rect = patches.Rectangle((c, r), 1, 1, linewidth=0.5,
                                     edgecolor='#DDDDDD', facecolor=color)  # Lighter grid lines
            ax.add_patch(rect)

    # --- Mark Start as a circle ---
    start_patch = patches.Circle((start[0] + 0.5, start[1] + 0.5), 0.3,
                                 facecolor='#33FF33', edgecolor='black', label='Start', zorder=10)
    ax.add_patch(start_patch)

    # --- Mark Goal as a star ---
    star_path = create_star((goal[0] + 0.5, goal[1] + 0.5))
    goal_patch = patches.PathPatch(star_path, facecolor='#FF3333',
                                   edgecolor='black', label='Goal', zorder=10)
    ax.add_patch(goal_patch)

    # --- Plot Paths ---
    # Define colors for color print
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Enhanced line styles for better black and white distinction
    linestyles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]
    markers = ['', 'o', '^', 's']  # Added markers for better distinction in black and white
    marker_sizes = [0, 4, 5, 4]  # Size of markers
    marker_intervals = [0, 20, 25, 15]  # Interval between markers (every Nth point)

    legend_elements = []
    algo_keys = list(paths_dict.keys())

    # Algorithm name abbreviations for more compact legend
    algo_abbr = {
        'A*': 'A*',
        'ACO': 'ACO',
        'GA': 'GA',
        'Hybrid A*-ACO': 'A*-ACO'
    }

    for i, algorithm in enumerate(algo_keys):
        path = paths_dict[algorithm]
        runtime_sec = run_times.get(algorithm, -1.0)
        runtime_ms = runtime_sec * 1000 if runtime_sec >= 0 else -1.0

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        linewidth = line_widths[i % len(line_widths)]
        marker = markers[i % len(markers)]
        marker_size = marker_sizes[i % len(marker_sizes)]
        marker_interval = marker_intervals[i % len(marker_intervals)]

        if path:
            # Calculate length as number of steps
            path_len_steps = len(path) - 1
            # Extract coordinates
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]

            # Create markevery parameter - only add markers periodically if marker is defined
            markevery = marker_interval if marker and marker_interval > 0 else None

            # Plot the path
            line, = ax.plot(path_x, path_y,
                            linestyle=linestyle,
                            color=color,
                            linewidth=linewidth,
                            marker=marker,
                            markersize=marker_size,
                            markevery=markevery,
                            alpha=0.85,
                            zorder=5 + i)

            # More compact legend label
            short_name = algo_abbr.get(algorithm, algorithm[:3])
            if runtime_ms >= 0:
                legend_label = f'{short_name}: {path_len_steps}步 ({runtime_ms:.1f}ms)'
            else:
                legend_label = f'{short_name}: {path_len_steps}步 (N/A)'

            # Create legend element
            legend_elements.append(
                plt.Line2D([0], [0],
                           color=color,
                           linestyle=linestyle,
                           marker=marker,
                           markersize=marker_size if marker else 0,
                           lw=3,
                           label=legend_label)
            )
        else:
            # For paths not found
            short_name = algo_abbr.get(algorithm, algorithm[:3])
            if runtime_ms >= 0:
                legend_label = f'{short_name}: 无路径 ({runtime_ms:.1f}ms)'
            else:
                legend_label = f'{short_name}: 无路径 (N/A)'

            legend_elements.append(
                plt.Line2D([0], [0],
                           color=colors[i % len(colors)],
                           linestyle=':',
                           lw=1,
                           label=legend_label)
            )

    # --- Configure Axes and Display ---
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, map_size + 1, 5))
    ax.set_yticks(np.arange(0, map_size + 1, 5))
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(False)

    # Add Legend with compact format
    ax.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              fancybox=True,
              shadow=False,
              ncol=min(4, len(paths_dict) + 1))

    # Add Title
    plt.title('联合算法与多种算法对比', pad=15, fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the figure
    try:
        plt.savefig(f'pathfinding_comparison_{map_size}x{map_size}.png', dpi=200, bbox_inches='tight')
        print(f"Visualization saved to pathfinding_comparison_{map_size}x{map_size}.png")
    except Exception as e:
        print(f"Error saving visualization: {e}")

    plt.show()  # Display the plot