# astar_aco_hybrid.py
import numpy as np
import math
import heapq
import random
import time # Keep time for potential profiling if needed

# --- Helper: Calculate path length with diagonal moves ---
def calculate_hybrid_path_length(path):
    """Calculate path length using Euclidean distance (sqrt(2) for diagonals)."""
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx == 1 and dy == 1:
            length += 1.41421356 # sqrt(2)
        elif dx == 1 or dy == 1: # Cardinal move
            length += 1.0
        # else: should not happen for adjacent steps
    return length

# --- Adapted A* Class (8 directions, Euclidean heuristic) ---
class AStarHybrid:
    """Adapted A* algorithm for the hybrid approach (8 directions)."""
    def __init__(self, start, goal, grid):
        self.s_start = start # (x, y)
        self.s_goal = goal   # (x, y)
        self.grid = grid # 2D list (0=obstacle, 1=free)
        self.map_height = len(grid)
        self.map_width = len(grid[0]) if self.map_height > 0 else 0

        # 8 directions (dx, dy)
        self.motions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                        (-1, 0), (-1, -1), (0, -1), (1, -1)]
        self.motion_costs = [1, 1.414, 1, 1.414, 1, 1.414, 1, 1.414] # Cost for each motion

        self.open_set = []  # priority queue (f_score, node)
        self.closed_set = set()
        self.g = {}  # g_score (cost from start)
        self.parent = {}  # parent node map
        self.h = {}  # heuristic cache (optional)

    def _is_valid(self, node):
        x, y = node
        return 0 <= x < self.map_width and 0 <= y < self.map_height and self.grid[y][x] == 1

    # Helper to check for diagonal obstacle blockage
    def _is_diagonal_blocked(self, current_node, neighbor_node):
        cx, cy = current_node
        nx, ny = neighbor_node
        dx = nx - cx
        dy = ny - cy

        # Only apply check for diagonal moves
        if abs(dx) == 1 and abs(dy) == 1:
            # Check the two adjacent cells
            cell1_x, cell1_y = cx + dx, cy
            cell2_x, cell2_y = cx, cy + dy

            # Check bounds for safety, though _is_valid should cover this for the neighbor
            # The cells (cell1_x, cell1_y) and (cell2_x, cell2_y) must be within bounds
            is_cell1_valid = 0 <= cell1_x < self.map_width and 0 <= cell1_y < self.map_height
            is_cell2_valid = 0 <= cell2_x < self.map_width and 0 <= cell2_y < self.map_height

            if is_cell1_valid and is_cell2_valid:
                 # Return True if *both* adjacent cells are obstacles
                 return self.grid[cell1_y][cell1_x] == 0 and self.grid[cell2_y][cell2_x] == 0
            # If adjacent cells are out of bounds, treat as not blocked diagonally
            # This scenario might need refinement depending on desired edge behavior
            return False
        return False # Not a diagonal move

    def _calculate_h(self, s):
        # Euclidean distance heuristic
        return math.sqrt((s[0] - self.s_goal[0]) ** 2 + (s[1] - self.s_goal[1]) ** 2)

    def search(self):
        """Performs the A* search."""
        self.g[self.s_start] = 0
        self.parent[self.s_start] = self.s_start
        h_start = self._calculate_h(self.s_start)
        f_start = self.g[self.s_start] + h_start
        heapq.heappush(self.open_set, (f_start, self.s_start))

        visited_nodes_for_plot = [] # Track visited for potential plotting later

        while self.open_set:
            current_f, current = heapq.heappop(self.open_set)

            # Check if already processed with lower f_score (if node added multiple times)
            if current in self.closed_set:
                 continue

            visited_nodes_for_plot.append(current)

            if current == self.s_goal:
                # Path found - reconstruct
                path = []
                temp = current
                while temp != self.s_start:
                    path.append(temp)
                    temp = self.parent[temp]
                path.append(self.s_start)
                return path[::-1], visited_nodes_for_plot # Return path and visited nodes

            self.closed_set.add(current)

            # Explore neighbors
            for i, motion in enumerate(self.motions):
                neighbor_x = current[0] + motion[0]
                neighbor_y = current[1] + motion[1]
                neighbor = (neighbor_x, neighbor_y)

                # Check basic validity and if already closed
                if not self._is_valid(neighbor) or neighbor in self.closed_set:
                    continue

                # --- Added Diagonal Obstacle Check ---
                if self._is_diagonal_blocked(current, neighbor):
                     continue
                # --- End Added Check ---

                cost = self.motion_costs[i]
                tentative_g = self.g[current] + cost

                if tentative_g < self.g.get(neighbor, float('inf')):
                    self.parent[neighbor] = current
                    self.g[neighbor] = tentative_g
                    h_neighbor = self._calculate_h(neighbor)
                    f_neighbor = tentative_g + h_neighbor
                    # Add to open set (duplicates are okay with heapq)
                    heapq.heappush(self.open_set, (f_neighbor, neighbor))

        print("A* Hybrid: No path found!")
        return [], visited_nodes_for_plot # No path found

# --- Adapted Ant Colony Class (8 directions) ---
class AntColonyHybrid:
    """Adapted Ant Colony Optimization for the hybrid approach (8 directions)."""
    def __init__(self, start, goal, grid, init_path=None):
        self.s_start = start # (x, y)
        self.s_goal = goal   # (x, y)
        self.grid = grid     # 2D list (0=obstacle, 1=free)
        self.map_height = len(grid)
        self.map_width = len(grid[0]) if self.map_height > 0 else 0

        # 8 directions (dx, dy) and costs
        self.motions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                        (-1, 0), (-1, -1), (0, -1), (1, -1)]
        self.motion_costs = [1, 1.414, 1, 1.414, 1, 1.414, 1, 1.414]
        self.num_directions = len(self.motions)

        # ACO parameters (tuned from Candd.py)
        self.ant_count = 50
        self.max_iterations = 150
        self.alpha = 1.5  # Pheromone influence
        self.beta = 3.0   # Heuristic influence
        self.rho = 0.2    # Evaporation rate
        self.Q = 10.0     # Pheromone deposit factor
        self.elite_ants = 3
        self.elite_factor = 3.0

        # Pheromone and heuristic matrices (height, width, directions)
        # Storing pheromone related to the edge *leaving* node (y, x) in direction i
        self.pheromone = np.ones((self.map_height, self.map_width, self.num_directions)) * 0.1
        self.heuristic = np.ones((self.map_height, self.map_width, self.num_directions)) * 1e-6 # Avoid zero div

        self.stagnation_counter = 0
        self.last_best_length = float('inf')
        self.stagnation_limit = 20

        self.best_path = None
        self.best_path_length = float('inf') # Uses hybrid length calculation
        self.all_visited_nodes = set() # Track all nodes visited by any ant

        self._initialize_heuristic()
        if init_path:
            self._enhance_pheromone_along_path(init_path)

    def _is_valid(self, node):
        x, y = node
        return 0 <= x < self.map_width and 0 <= y < self.map_height and self.grid[y][x] == 1

    # Helper to check for diagonal obstacle blockage (same logic as AStarHybrid)
    def _is_diagonal_blocked(self, current_node, neighbor_node):
        cx, cy = current_node
        nx, ny = neighbor_node
        dx = nx - cx
        dy = ny - cy

        # Only apply check for diagonal moves
        if abs(dx) == 1 and abs(dy) == 1:
            # Check the two adjacent cells
            cell1_x, cell1_y = cx + dx, cy
            cell2_x, cell2_y = cx, cy + dy

            # Check bounds for safety
            is_cell1_valid = 0 <= cell1_x < self.map_width and 0 <= cell1_y < self.map_height
            is_cell2_valid = 0 <= cell2_x < self.map_width and 0 <= cell2_y < self.map_height

            if is_cell1_valid and is_cell2_valid:
                 # Return True if *both* adjacent cells are obstacles
                 return self.grid[cell1_y][cell1_x] == 0 and self.grid[cell2_y][cell2_x] == 0
            return False
        return False # Not a diagonal move


    def _initialize_heuristic(self):
        """Initialize heuristic based on inverse Euclidean distance to goal."""
        for r in range(self.map_height):
            for c in range(self.map_width):
                if self.grid[r][c] == 1: # If current node (c, r) is walkable
                    current_pos = (c, r)
                    for i, motion in enumerate(self.motions):
                        next_x, next_y = c + motion[0], r + motion[1]
                        next_pos = (next_x, next_y)

                        # Check basic validity and diagonal blockage for heuristic calculation
                        if self._is_valid(next_pos) and not self._is_diagonal_blocked(current_pos, next_pos):
                            dist_to_goal = math.sqrt((next_x - self.s_goal[0])**2 + (next_y - self.s_goal[1])**2)
                            self.heuristic[r, c, i] = 1.0 / (dist_to_goal + 0.1) # Store at (r, c) for direction i
                        else:
                            # If not valid or diagonally blocked, set heuristic to a very small value
                            self.heuristic[r, c, i] = 1e-9


    def _enhance_pheromone_along_path(self, path):
        """Increase pheromone along the provided initial path (e.g., from A*)."""
        if not path: return
        path_length_steps = len(path)

        for i in range(path_length_steps - 1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            dx, dy = x2 - x1, y2 - y1

            # Dynamic enhancement factor (stronger near start/end)
            # position_factor = max(5.0 * (1 - i / path_length_steps), 5.0 * (i / path_length_steps))
            enhancement = 15.0 # Simpler constant enhancement

            # Find direction index
            try:
                idx = self.motions.index((dx, dy))
                # Enhance pheromone on the edge leaving (x1, y1)
                self.pheromone[y1, x1, idx] = max(self.pheromone[y1, x1, idx], enhancement)

                # Optional: Enhance nearby directions slightly (from original)
                # for nearby_idx in range(self.num_directions):
                #     if nearby_idx != idx: self.pheromone[y1, x1, nearby_idx] *= 1.2
            except ValueError:
                print(f"Warning: Motion {(dx, dy)} not found in motions list during pheromone enhancement.")
                pass # Should not happen if path consists of valid motions

        # Optional: Add small random variations (from original)
        # self.pheromone *= (1.0 + np.random.rand(*self.pheromone.shape) * 0.1)


    def _select_next_node(self, ant_position, tabu_list):
        """Select the next node based on pheromone, heuristic, and randomness."""
        x, y = ant_position # Current (col, row)
        probabilities = []
        possible_moves = [] # Store (next_pos, direction_index)
        total_prob = 0.0

        for i, motion in enumerate(self.motions):
            next_x, next_y = x + motion[0], y + motion[1]
            next_pos = (next_x, next_y)

            # Check basic validity, diagonal blockage, and if visited in path
            if self._is_valid(next_pos) and not self._is_diagonal_blocked(ant_position, next_pos) and next_pos not in tabu_list:
                tau = self.pheromone[y, x, i]
                eta = self.heuristic[y, x, i]
                prob = (tau ** self.alpha) * (eta ** self.beta)
                probabilities.append(prob)
                possible_moves.append((next_pos, i))
                total_prob += prob

        if not possible_moves:
            return None, -1 # No valid moves

        if total_prob <= 1e-9: # Handle zero total probability
            # Fallback: choose a random valid move
             chosen_idx = random.choice(range(len(possible_moves)))
        else:
            probabilities = np.array(probabilities) / total_prob
            chosen_idx = np.random.choice(len(possible_moves), p=probabilities)

        return possible_moves[chosen_idx] # Returns (next_pos, direction_index)

    def _construct_ant_path(self):
        """Simulates one ant finding a path."""
        current_pos = self.s_start
        path = [current_pos]
        tabu_list = {current_pos} # Use set for faster lookup
        max_steps = self.map_width * self.map_height * 2 # Increased limit slightly

        while current_pos != self.s_goal and len(path) <= max_steps:
            next_move_result = self._select_next_node(current_pos, tabu_list)

            if next_move_result is None or next_move_result[0] is None:
                # Ant got stuck
                return None # Indicate failure

            next_pos, _ = next_move_result
            path.append(next_pos)
            tabu_list.add(next_pos)
            self.all_visited_nodes.add(next_pos) # Track all visited nodes globally
            current_pos = next_pos

        if current_pos == self.s_goal:
            return path
        else:
            # Did not reach goal (stuck or max steps exceeded)
            return None


    def _update_pheromone(self, ant_paths_and_lengths):
        """Update pheromone based on the paths found in one iteration."""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        self.pheromone = np.maximum(self.pheromone, 0.01) # Min pheromone level

        # Sort paths by quality (length) - shorter is better
        ant_paths_and_lengths.sort(key=lambda x: x[1])

        # Pheromone deposit for all successful ants
        for i, (path, length) in enumerate(ant_paths_and_lengths):
            deposit = self.Q / length if length > 0 else self.Q # Base deposit

            # Elite bonus
            if i < self.elite_ants:
                 deposit *= self.elite_factor #* (self.elite_ants - i + 1) # Weighted elite bonus

            for step in range(len(path) - 1):
                x1, y1 = path[step]
                x2, y2 = path[step + 1]
                dx, dy = x2 - x1, y2 - y1
                try:
                    idx = self.motions.index((dx, dy))
                    # Deposit pheromone on the edge leaving (x1, y1)
                    self.pheromone[y1, x1, idx] += deposit
                except ValueError:
                     pass # Should not happen for valid paths

        # Optional: Extra deposit for the globally best path found so far
        if self.best_path and self.best_path_length != float('inf'):
            best_deposit = self.Q / self.best_path_length * self.elite_factor * 1.5 # Stronger bonus for best-so-far
            for step in range(len(self.best_path) - 1):
                 x1, y1 = self.best_path[step]
                 x2, y2 = self.best_path[step+1]
                 dx, dy = x2-x1, y2-y1
                 try:
                    idx = self.motions.index((dx,dy))
                    self.pheromone[y1,x1,idx] += best_deposit
                 except ValueError:
                     pass


    def run(self):
        """Runs the ACO algorithm."""
        print(f"Starting ACO Hybrid ({self.ant_count} ants, {self.max_iterations} iterations)")
        start_time = time.time()

        for iteration in range(self.max_iterations):
            iteration_paths_lengths = [] # Store (path, hybrid_length) for this iteration

            for ant in range(self.ant_count):
                path = self._construct_ant_path()
                if path:
                    # Path smoothing could be added here if desired
                    # path = self._path_smoothing(path) # Needs implementation
                    length = calculate_hybrid_path_length(path)
                    iteration_paths_lengths.append((path, length))

                    # Update global best path if this one is better
                    if length < self.best_path_length:
                        self.best_path = path # Store copy if needed
                        self.best_path_length = length
                        print(f"  Iter {iteration}: New best path length: {length:.2f}")
                        self.stagnation_counter = 0 # Reset stagnation on improvement

            # Update pheromone based on paths found in this iteration
            if iteration_paths_lengths:
                 self._update_pheromone(iteration_paths_lengths)
            else:
                 # Handle case where no ants found a path this iteration
                 self.stagnation_counter +=1
                 # Aggressive evaporation if no paths found? Maybe not needed.

            # Check stagnation
            if self.best_path_length == self.last_best_length:
                 self.stagnation_counter += 1
            else:
                 self.stagnation_counter = 0 # Reset if best length changed
            self.last_best_length = self.best_path_length

            if self.stagnation_counter >= self.stagnation_limit:
                 print(f"ACO Hybrid: Early stopping at iteration {iteration} due to stagnation.")
                 break

            if iteration % 20 == 0 and iteration > 0:
                 print(f"  Iter {iteration}: Current best length: {self.best_path_length:.2f}")

        end_time = time.time()
        print(f"ACO Hybrid finished in {end_time - start_time:.2f} seconds.")

        if self.best_path:
            print(f"ACO Hybrid final best path length: {self.best_path_length:.2f}")
            return self.best_path, list(self.all_visited_nodes)
        else:
            print("ACO Hybrid failed to find any path.")
            return [], list(self.all_visited_nodes)


# --- Main Hybrid A*-ACO Class ---
class AStarACOHybridRunner:
    """Orchestrates the hybrid A*-ACO algorithm."""
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.map_height = len(grid)
        self.map_width = len(grid[0]) if self.map_height > 0 else 0

    def run(self):
        """Runs the hybrid algorithm."""
        print("\n--- Running Hybrid A*-ACO ---")
        # 1. Run initial A* (using the hybrid 8-directional version with corner cutting restriction)
        print("Step 1: Running initial A* (8 directions, no diagonal obstacles)...")
        astar_hybrid = AStarHybrid(self.start, self.goal, self.grid)
        astar_path, _ = astar_hybrid.search() # Don't need visited nodes from A* here

        if not astar_path:
            print("Hybrid A*-ACO Error: Initial A* failed to find a path.")
            return [] # Return empty list if A* fails

        astar_length = calculate_hybrid_path_length(astar_path)
        print(f"Initial A* path length: {astar_length:.2f} (using sqrt(2) for diagonals)")

        # 2. Run ACO, initialized with the A* path (using 8 directions with corner cutting restriction)
        print("\nStep 2: Running ACO initialized with A* path (8 directions, no diagonal obstacles)...")
        aco_hybrid = AntColonyHybrid(self.start, self.goal, self.grid, init_path=astar_path)
        aco_path, _ = aco_hybrid.run() # Don't need visited nodes from ACO for comparison plot

        # 3. Determine final path (usually ACO's result if successful)
        if aco_path:
            aco_length = calculate_hybrid_path_length(aco_path)
            print(f"Final ACO path length: {aco_length:.2f}")
            improvement = ((astar_length - aco_length) / astar_length) * 100 if astar_length > 0 else 0
            if improvement > 0.1: # Threshold for meaningful improvement
                 print(f"ACO improved path by {improvement:.2f}% compared to initial A*.")
            elif improvement < -0.1:
                 print(f"Warning: ACO path is {-improvement:.2f}% longer than initial A* path.")

            final_path = aco_path
        else:
            print("ACO failed; falling back to the initial A* path.")
            final_path = astar_path # Fallback to A* result

        print("--- Hybrid A*-ACO Finished ---")
        return final_path