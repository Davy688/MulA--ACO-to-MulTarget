"""
@author: Dz-ZiYang Deng
"""
import heapq
import math
import random
import numpy as np


def euclidean_distance(a, b):
    """
    Args:
        a: First point (x, y)
        b: Second point (x, y)
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def euclidean_distance_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def is_diagonal_valid(r, c, nr, nc, grid):
    return grid[r, nc] == 0 and grid[nr, c] == 0


def get_neighbors(node, rows, cols, grid):
    """
    Args:
        node: Current node (row, col)
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        grid: The grid (2D numpy array)
    """
    r, c = node
    neighbors = []

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr, nc] == 0:
                if abs(dr) == 1 and abs(dc) == 1:
                    if not is_diagonal_valid(r, c, nr, nc, grid):
                        continue
                neighbors.append((nr, nc))

    return neighbors


def heuristic(a, b):
    """
    Args:
        a: First node
        b: Second node
    """
    return euclidean_distance(a, b)


def a_star_search(start, goal, rows, cols, grid, config_params):
    """
    Args:
        start: Start node (row, col)
        goal: Goal node (row, col)
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        grid: The grid (2D numpy array)
        config_params: Dictionary of configuration parameters
    """
    if not start or not goal:
        return []

    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []

    open_set = []
    closed_set = set()

    g_score = {start: 0}
    parent = {}

    heapq.heappush(open_set, (heuristic(start, goal), start))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)

        for neighbor in get_neighbors(current, rows, cols, grid):
            if neighbor in closed_set:
                continue

            dr = neighbor[0] - current[0]
            dc = neighbor[1] - current[1]

            move_cost = euclidean_distance(current, neighbor)
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = current

                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return []


# --- MultiA* Algorithm (Multi-Modal A*) ---

def mm_star_search(start, goal, waypoints, rows, cols, grid, config_params):
    """
    Args:
        start: Start node (row, col)
        goal: Goal node (row, col)
        waypoints: List of intermediate waypoints [(row, col)]
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        grid: The grid (2D numpy array)
        config_params: Dictionary of configuration parameters
    """
    if not start or not goal:
        return []

    if not waypoints:
        return a_star_search(start, goal, rows, cols, grid, config_params)

    all_points = [start] + waypoints + [goal]

    final_path = []

    for i in range(len(all_points) - 1):
        from_point = all_points[i]
        to_point = all_points[i + 1]

        segment = a_star_search(from_point, to_point, rows, cols, grid, config_params)

        if not segment:
            print(f"Failed to find path from {from_point} to {to_point}")
            return []

        if i == 0:
            final_path.extend(segment)
        else:
            final_path.extend(segment[1:])

    return final_path


def ant_colony_optimization(base_path, rows, cols, grid, config_params):
    """
    Args:
        base_path: Initial path to optimize [(row, col)]
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        grid: The grid (2D numpy array)
        config_params: Dictionary of ACO parameters
    """
    if not base_path or len(base_path) < 2:
        return base_path

    num_ants = config_params.get('ACO_NUM_ANTS', 40)
    iterations = config_params.get('ACO_ITERATIONS', 30)
    alpha = config_params.get('ACO_ALPHA', 1.0)
    beta = config_params.get('ACO_BETA', 2.5)
    rho = config_params.get('ACO_RHO', 0.2)
    q = config_params.get('ACO_Q', 100)
    initial_pheromone = config_params.get('ACO_INITIAL_PHEROMONE', 0.1)
    min_pheromone = config_params.get('ACO_PHEROMONE_MIN', 0.01)
    max_pheromone = config_params.get('ACO_PHEROMONE_MAX', 10.0)

    pheromone = {}
    for i in range(len(base_path) - 1):
        edge = (base_path[i], base_path[i + 1])
        pheromone[edge] = initial_pheromone

    best_path = list(base_path)
    best_length = sum(euclidean_distance(base_path[i], base_path[i + 1])
                      for i in range(len(base_path) - 1))

    for iteration in range(iterations):
        ant_paths = []
        ant_lengths = []

        for ant in range(num_ants):
            current = base_path[0]
            ant_path = [current]
            visited = {current}

            while current != base_path[-1]:
                neighbors = get_neighbors(current, rows, cols, grid)

                probabilities = []

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    edge = (current, neighbor)
                    edge_pheromone = pheromone.get(edge, initial_pheromone)

                    distance = euclidean_distance(neighbor, base_path[-1])
                    if distance == 0:
                        distance = 0.1

                    prob = (edge_pheromone ** alpha) * ((1.0 / distance) ** beta)
                    probabilities.append((neighbor, prob))

                if not probabilities:
                    break

                total = sum(p for _, p in probabilities)
                if total == 0:
                    next_node = random.choice([n for n, _ in probabilities])
                else:
                    # Roulette wheel selection
                    r = random.random() * total
                    cumsum = 0
                    next_node = probabilities[0][0]  # Default
                    for node, prob in probabilities:
                        cumsum += prob
                        if cumsum >= r:
                            next_node = node
                            break

                # Move to next node
                current = next_node
                ant_path.append(current)
                visited.add(current)

                # If we're stuck in a loop or path is too long, break
                if len(ant_path) > 3 * len(base_path):
                    break

            # Calculate path length
            path_length = sum(euclidean_distance(ant_path[i], ant_path[i + 1])
                              for i in range(len(ant_path) - 1))

            ant_paths.append(ant_path)
            ant_lengths.append(path_length)

            # Update best path if this ant found a shorter one
            if path_length < best_length:
                best_path = list(ant_path)
                best_length = path_length

        # Update pheromones
        # First, evaporate all pheromones
        for edge in pheromone:
            pheromone[edge] *= (1 - rho)
            pheromone[edge] = max(min_pheromone, min(max_pheromone, pheromone[edge]))

        # Then, add new pheromones based on ant paths
        for path, length in zip(ant_paths, ant_lengths):
            # Skip invalid paths
            if len(path) < 2:
                continue

            # Amount of pheromone to deposit depends on path quality
            deposit = q / length

            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                pheromone[edge] = pheromone.get(edge, 0) + deposit
                pheromone[edge] = max(min_pheromone, min(max_pheromone, pheromone[edge]))
    print(f"[DEBUG] ACO Best Path Length: {len(best_path)}")
    print(f"[DEBUG] ACO Best Path: {best_path}")

    # Apply path smoothing to the best path found
    return smooth_path(best_path, grid, rows, cols, config_params)


# --- Path Smoothing ---

def smooth_path(path, grid, rows, cols, config_params):
    """
    Args:
        path: The path to smooth [(row, col)]
        grid: The grid (2D numpy array)
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        config_params: Dictionary of smoothing parameters
    """
    if not path or len(path) <= 2:
        return path

    # Extract smoothing parameters
    iterations = config_params.get('SMOOTHING_ITERATIONS', 50)
    alpha = config_params.get('SMOOTHING_ALPHA', 0.3)  # Increased from 0.2 to 0.3
    beta = config_params.get('SMOOTHING_BETA', 0.3)  # Increased from 0.2 to 0.3

    smoothed_path = list(path)

    for _ in range(iterations):
        for i in range(1, len(smoothed_path) - 1):
            # Current point
            r, c = smoothed_path[i]

            # Previous and next points
            prev_r, prev_c = smoothed_path[i - 1]
            next_r, next_c = smoothed_path[i + 1]

            # Calculate new position using weighted average
            new_r = r + alpha * (prev_r - r) + beta * (next_r - r)
            new_c = c + alpha * (prev_c - c) + beta * (next_c - c)

            # Convert to grid coordinates
            new_r_int = int(round(new_r))
            new_c_int = int(round(new_c))

            # Skip if position didn't change
            if new_r_int == r and new_c_int == c:
                continue

            # Check if new position is valid (not an obstacle)
            if (0 <= new_r_int < rows and 0 <= new_c_int < cols and
                    grid[new_r_int, new_c_int] == 0):
                # For diagonal moves, check that both orthogonal cells are free
                r_int, c_int = int(round(r)), int(round(c))

                # If this is a diagonal move from the previous position
                if abs(new_r_int - r_int) == 1 and abs(new_c_int - c_int) == 1:
                    # Check if the diagonal move is valid
                    if not is_diagonal_valid(r_int, c_int, new_r_int, new_c_int, grid):
                        continue  # Skip this smoothing step

                # Update position
                smoothed_path[i] = (new_r_int, new_c_int)

    # Remove duplicate consecutive points
    i = 0
    while i < len(smoothed_path) - 1:
        if smoothed_path[i] == smoothed_path[i + 1]:
            smoothed_path.pop(i + 1)
        else:
            i += 1

    return smoothed_path


# --- Optional Motion Model (for DWA, if needed) ---

def motion_model_continuous(x, y, theta, v, w, dt):
    """
    Args:
        x: Current x position
        y: Current y position
        theta: Current orientation (radians)
        v: Linear velocity
        w: Angular velocity
        dt: Time step
    """
    # For circular motion
    if abs(w) > 0.001:
        # Radius of curvature
        r = v / w
        # New position and orientation
        theta_new = theta + w * dt
        x_new = x + r * (math.sin(theta_new) - math.sin(theta))
        y_new = y - r * (math.cos(theta_new) - math.cos(theta))
    else:
        # Straight line motion
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta

    return x_new, y_new, theta_new