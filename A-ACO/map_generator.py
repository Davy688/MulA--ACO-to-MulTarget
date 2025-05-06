# map_generator.py
import numpy as np
import random
import math
from collections import deque

def generate_map(map_size=20, seed=None):
    """
    Generates an obstacle map with a density gradient favoring paths through the middle.

    Args:
        map_size (int): The width and height of the square map.
        seed (int, optional): Random seed for reproducibility. If None, uses a random seed.

    Returns:
        tuple: (grid, obstacle_set)
            grid (list[list[int]]): 2D list representing the map (0=obstacle, 1=free).
            obstacle_set (set[tuple[int, int]]): Set containing coordinates (x, y) of obstacles.
            Returns (None, None) if map generation fails repeatedly.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    print(f"Generating map with size {map_size}x{map_size} using seed {seed}...")

    max_retries = 10
    for attempt in range(max_retries):
        grid = [[1] * map_size for _ in range(map_size)] # Initialize all as free
        obstacle_grid_bool = [[False] * map_size for _ in range(map_size)] # For neighbor check
        obstacle_set = set()

        # Parameters for density gradient
        base_density = 0.28
        center_x, center_y = (map_size - 1) / 2.0, (map_size - 1) / 2.0 # Use float center
        max_dist = math.sqrt(center_x**2 + center_y**2)
        min_density_factor = 0.3
        max_density_factor = 1.8

        # Safe zones near start (0,0) and end (map_size-1, map_size-1)
        safe_radius = 2
        start_x, start_y = 0, 0
        goal_x, goal_y = map_size - 1, map_size - 1

        for r in range(map_size): # row (y)
            for c in range(map_size): # col (x)
                # Keep start and end areas clear
                if math.sqrt((c - start_x)**2 + (r - start_y)**2) <= safe_radius or \
                   math.sqrt((c - goal_x)**2 + (r - goal_y)**2) <= safe_radius:
                    grid[r][c] = 1
                    obstacle_grid_bool[r][c] = False
                    continue

                # Calculate density based on distance from center
                dist_from_center = math.sqrt((c - center_x)**2 + (r - center_y)**2)
                norm_dist = dist_from_center / max_dist if max_dist > 0 else 0
                density_factor = min_density_factor + (max_density_factor - min_density_factor) * (norm_dist**1.5)
                current_density = min(max(base_density * density_factor, 0.05), 0.7)

                # Check neighbors to avoid clumping
                nearby_obstacles = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < map_size and 0 <= nc < map_size and obstacle_grid_bool[nr][nc]:
                            nearby_obstacles += 1

                # Place obstacle
                if random.random() < current_density and nearby_obstacles < 4:
                    grid[r][c] = 0
                    obstacle_grid_bool[r][c] = True
                    obstacle_set.add((c, r)) # Add (x, y) to set

        # --- Connectivity Check (BFS) ---
        q = deque([(start_x, start_y)]) # Start at (x,y) = (0,0)
        visited = set([(start_x, start_y)])
        reachable = False
        if grid[start_y][start_x] == 0: # Check if start is blocked
             print(f"Attempt {attempt + 1}: Start node is blocked. Regenerating...")
             continue

        while q:
            curr_x, curr_y = q.popleft()
            if curr_x == goal_x and curr_y == goal_y:
                reachable = True
                break
            # Explore neighbors (4-directional for basic connectivity check)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < map_size and 0 <= ny < map_size and \
                   grid[ny][nx] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        if reachable:
            print(f"Successfully generated valid map (Size: {map_size}x{map_size}, Seed: {seed}).")
            return grid, obstacle_set
        else:
            print(f"Attempt {attempt + 1}: Map generation failed connectivity check. Retrying...")

    print(f"Error: Failed to generate a valid map after {max_retries} attempts.")
    return None, None