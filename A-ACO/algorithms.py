# algorithms.py
import numpy as np
import random
import heapq
import math
from collections import deque

# --- Standard A* Algorithm (4 directions, Manhattan heuristic) ---
def astar_standard(grid, start, goal):
    """Standard A* algorithm implementation (4 directions)."""
    map_height = len(grid)
    map_width = len(grid[0]) if map_height > 0 else 0

    def heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4 directions (dx, dy)

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = []
    heapq.heappush(open_set, (fscore[start], start))

    while open_set:
        current_f, current = heapq.heappop(open_set)

        # Optimization: If we find a shorter path to a node already processed, ignore
        if current in close_set:# and current_f > fscore[current]: # Check if actually needed
             continue

        if current == goal:
            path = []
            temp = current
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            path.append(start)
            return path[::-1] # Return reversed path

        close_set.add(current)

        for dx, dy in neighbors:
            neighbor_x, neighbor_y = current[0] + dx, current[1] + dy
            neighbor = (neighbor_x, neighbor_y)

            # Check bounds and obstacles
            if not (0 <= neighbor_x < map_width and 0 <= neighbor_y < map_height and grid[neighbor_y][neighbor_x] == 1):
                 continue

            tentative_gscore = gscore[current] + 1 # Cost is 1 for 4-directional moves

            if tentative_gscore < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)
                # Check if neighbor is in open_set before adding
                # Simple add works with heapq but might add duplicates with different priorities
                heapq.heappush(open_set, (fscore[neighbor], neighbor))

    print("A* Standard: No path found!")
    return []

# --- Standard Ant Colony Optimization (4 directions) ---
def ant_colony_standard(grid, start, goal, n_ants=30, n_iterations=50, alpha=1.0, beta=3.0, evaporation_rate=0.6, q=100):
    """Standard Ant Colony Optimization (4 directions)."""
    map_height = len(grid)
    map_width = len(grid[0]) if map_height > 0 else 0
    map_size = map_width # Assuming square map for simplicity in pheromone indexing if needed elsewhere
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)] # E, S, W, N (dx, dy)
    neighbor_indices = { (0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3 } # Map (dx, dy) to index

    # Initialize pheromone matrix (map_height, map_width, 4 directions)
    # Store pheromone on the node *from* which the move originates
    pheromone = np.ones((map_height, map_width, 4)) * 0.1

    # Heuristic information (inverse Manhattan distance to goal)
    heuristic_info = np.zeros((map_height, map_width, 4))
    for r in range(map_height):
        for c in range(map_width):
            if grid[r][c] == 1: # Only for walkable nodes
                for i, (dx, dy) in enumerate(neighbors):
                    nx, ny = c + dx, r + dy
                    if 0 <= nx < map_width and 0 <= ny < map_height and grid[ny][nx] == 1:
                        dist = abs(goal[0] - nx) + abs(goal[1] - ny)
                        heuristic_info[r, c, i] = 1.0 / (dist + 1e-6) # Use (r, c) index

    best_path = None
    best_path_length = float('inf')

    for iteration in range(n_iterations):
        all_paths = []
        all_lengths = []

        for ant in range(n_ants):
            current = start
            path = [current]
            visited_in_path = {current} # Avoid cycles within a single ant's path
            stuck = False
            max_steps = map_width * map_height # Limit steps

            while current != goal and len(path) <= max_steps:
                x, y = current # Current position (col, row)
                possible_moves = []
                probabilities = []
                total_prob = 0.0

                for i, (dx, dy) in enumerate(neighbors):
                    nx, ny = x + dx, y + dy
                    neighbor_pos = (nx, ny)

                    if 0 <= nx < map_width and 0 <= ny < map_height and \
                       grid[ny][nx] == 1 and neighbor_pos not in visited_in_path:

                        # Use pheromone[y, x, i] for move from (x,y) in direction i
                        tau = pheromone[y, x, i]
                        eta = heuristic_info[y, x, i]
                        prob = (tau ** alpha) * (eta ** beta)
                        possible_moves.append(neighbor_pos)
                        probabilities.append(prob)
                        total_prob += prob

                if not possible_moves:
                    stuck = True
                    break # Ant is stuck

                if total_prob > 0:
                    probabilities = np.array(probabilities) / total_prob
                    chosen_index = np.random.choice(len(possible_moves), p=probabilities)
                else: # Random choice if probabilities sum to 0
                    chosen_index = random.choice(range(len(possible_moves)))

                next_pos = possible_moves[chosen_index]
                path.append(next_pos)
                visited_in_path.add(next_pos)
                current = next_pos

            if not stuck and current == goal:
                path_length = len(path) - 1
                all_paths.append(path)
                all_lengths.append(path_length)
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

        # Evaporation
        pheromone *= (1 - evaporation_rate)
        pheromone = np.maximum(pheromone, 0.01) # Pheromone floor

        # Deposit pheromone
        for path, length in zip(all_paths, all_lengths):
            if length > 0:
                delta_pheromone = q / length
                for i in range(len(path) - 1):
                    x1, y1 = path[i]
                    x2, y2 = path[i + 1]
                    dx, dy = x2 - x1, y2 - y1
                    if (dx, dy) in neighbor_indices:
                        direction_index = neighbor_indices[(dx, dy)]
                        # Deposit pheromone on the starting node of the edge
                        pheromone[y1, x1, direction_index] += delta_pheromone


        # Optional: Elite ant strategy (add extra pheromone for the best path found so far)
        if best_path and best_path_length != float('inf'):
            delta_pheromone_elite = q / best_path_length * 0.5 # Elite bonus
            for i in range(len(best_path) - 1):
                 x1, y1 = best_path[i]
                 x2, y2 = best_path[i+1]
                 dx, dy = x2 - x1, y2 - y1
                 if (dx, dy) in neighbor_indices:
                     direction_index = neighbor_indices[(dx, dy)]
                     pheromone[y1, x1, direction_index] += delta_pheromone_elite

    if not best_path:
         print("ACO Standard: No path found!")
    return best_path if best_path else []



# --- 遗传算法 (4 directions) ---
def genetic_algorithm_standard(grid, start, goal, population_size=100, generations=50, mutation_rate=0.2,
                               elite_size=20):
    """Genetic Algorithm for pathfinding (4 directions)."""
    map_height = len(grid)
    map_width = len(grid[0]) if map_height > 0 else 0
    max_path_length = map_width * map_height // 2  # 降低了最大路径长度以提高效率

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, S, W, N (dx, dy)

    def decode_path(individual):
        """将基因序列解码为实际路径"""
        path = [start]
        current_x, current_y = start
        valid = True
        reached_goal = False

        for move_index in individual:
            if not 0 <= move_index < len(directions):
                valid = False
                break

            dx, dy = directions[move_index]
            nx, ny = current_x + dx, current_y + dy

            if not (0 <= nx < map_width and 0 <= ny < map_height and grid[ny][nx] == 1):
                valid = False
                break

            path.append((nx, ny))
            current_x, current_y = nx, ny

            if (nx, ny) == goal:
                reached_goal = True
                break  # 找到目标，立即结束路径

        return path, valid, reached_goal

    def calculate_fitness(individual):
        """计算个体适应度 - 值越小越好"""
        path, valid, reached_goal = decode_path(individual)

        if reached_goal:
            # 如果找到了目标，适应度就是路径长度（步数）
            return len(path) - 1

        # 否则，计算适应度基于以下几个因素：
        # 1. 有效移动的数量 (奖励更多有效移动)
        # 2. 最后有效位置到目标的曼哈顿距离 (惩罚距离目标远的路径)
        # 3. 无效移动的数量 (惩罚无效移动)

        last_valid_pos = path[-1] if valid else path[-1] if path else start
        valid_moves = len(path) - 1

        # 距离惩罚
        manhattan_dist = abs(last_valid_pos[0] - goal[0]) + abs(last_valid_pos[1] - goal[1])

        # 复合适应度：距离惩罚 + 无效移动惩罚 - 有效移动奖励
        return manhattan_dist * 5 + (len(individual) - valid_moves) * 10 - valid_moves * 0.5

    def create_individual():
        """创建一个新的个体（基因序列）"""
        # 估计一个合理的初始路径长度
        initial_len = abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + random.randint(5, max(10, (
                    map_width + map_height) // 4))
        path_len = min(max(initial_len, 10), max_path_length)
        return [random.randint(0, 3) for _ in range(path_len)]

    def crossover(parent1, parent2):
        """执行双点交叉操作"""
        if len(parent1) <= 2 or len(parent2) <= 2:
            return parent1[:], parent2[:]

        # 确保交叉点在有效范围内
        point1 = random.randint(1, min(len(parent1) - 1, len(parent2) - 1))
        point2 = random.randint(point1, min(len(parent1), len(parent2)))

        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2

    def mutate(individual):
        """对个体执行多种可能的变异"""
        if not individual:
            return individual

        mutated = individual[:]

        # 多种可能的变异类型
        mutation_type = random.random()

        if mutation_type < 0.4:  # 修改一个移动方向
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = random.randint(0, 3)

        elif mutation_type < 0.7 and len(mutated) < max_path_length:  # 插入一个新移动
            idx = random.randint(0, len(mutated))
            mutated.insert(idx, random.randint(0, 3))

        elif mutation_type < 0.9 and len(mutated) > 1:  # 删除一个移动
            idx = random.randint(0, len(mutated) - 1)
            del mutated[idx]

        else:  # 替换序列的一部分
            if len(mutated) >= 3:
                start_idx = random.randint(0, len(mutated) - 3)
                length = random.randint(2, min(5, len(mutated) - start_idx))
                new_segment = [random.randint(0, 3) for _ in range(length)]
                mutated[start_idx:start_idx + length] = new_segment

        return mutated

    def selection(population, fitness_scores):
        """使用锦标赛选择"""
        selected = []
        tournament_size = 5

        for _ in range(population_size):
            # 随机选择几个参赛者
            competitors = random.sample(range(len(population)), tournament_size)
            # 选择其中适应度最低的（最好的）
            winner_idx = min(competitors, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx])

        return selected

    # 初始化种群
    population = [create_individual() for _ in range(population_size)]
    best_individual = None
    best_fitness = float('inf')
    best_path = None
    generations_no_improvement = 0

    print("Starting GA with population size:", population_size)

    for generation in range(generations):
        # 计算适应度
        fitness_scores = [calculate_fitness(ind) for ind in population]

        # 找出当前最佳个体
        current_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx]
            path, valid, reached_goal = decode_path(best_individual)

            if reached_goal:
                best_path = path
                print(f"Generation {generation}: Found valid path to goal with length {len(best_path) - 1}")

                # 如果找到了足够好的路径，可以提前终止
                if len(best_path) - 1 <= abs(start[0] - goal[0]) + abs(start[1] - goal[1]) * 1.5:
                    print("Found near-optimal path, terminating early")
                    break

            generations_no_improvement = 0
        else:
            generations_no_improvement += 1

        # 如果连续多代没有改进，增加变异率来跳出局部最优
        if generations_no_improvement >= 10:
            mutation_rate = min(0.5, mutation_rate * 1.2)
            generations_no_improvement = 0

        # 精英保留
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_size]
        elites = [population[i] for i in elite_indices]

        # 选择
        selected = selection(population, fitness_scores)

        # 创建新一代
        next_gen = elites[:]  # 首先添加精英

        while len(next_gen) < population_size:
            # 选择父代
            parent1, parent2 = random.sample(selected, 2)

            # 交叉
            if random.random() < 0.8:  # 交叉概率
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # 变异
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            # 添加到下一代
            if len(next_gen) < population_size:
                next_gen.append(child1)
            if len(next_gen) < population_size:
                next_gen.append(child2)

        population = next_gen

        # 每隔几代输出一次状态
        if generation % 10 == 0:
            valid_paths = 0
            for ind in population:
                _, _, reached = decode_path(ind)
                if reached:
                    valid_paths += 1
            print(
                f"Generation {generation}: Best fitness = {best_fitness}, Valid paths: {valid_paths}/{population_size}")

    # 寻找可行的最佳路径
    if best_path:
        print(f"GA Standard: Final best path length = {len(best_path) - 1}")
        return best_path
    else:
        # 如果没有找到通向目标的路径，尝试找到最接近目标的路径
        print("GA Standard: Failed to find a complete path to goal, returning best attempt...")
        if best_individual:
            last_try_path, _, _ = decode_path(best_individual)
            return last_try_path
        return []