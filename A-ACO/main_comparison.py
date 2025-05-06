# main_comparison.py
import time
import argparse

# Import functions and classes from other files
from map_generator import generate_map
from algorithms import astar_standard, ant_colony_standard, genetic_algorithm_standard
from astar_aco_hybrid import AStarACOHybridRunner  # Import the runner class
from visualization import visualize_paths_single_map, visualize_map_only


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pathfinding Algorithm Comparison")
    parser.add_argument("--map-only", action="store_true",
                        help="Only visualize the map without running algorithms")
    parser.add_argument("--map-size", type=int, default=25,
                        help="Size of the map grid (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for map generation (default: 42, None for random)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the map-only visualization")
    return parser.parse_args()


# === Main Execution ===
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Configuration from arguments
    MAP_SIZE = args.map_size
    MAP_SEED = args.seed
    START_NODE = (0, 0)  # Start at top-left (x, y)

    # Algorithm parameters (can be tuned further)
    ACO_ANTS = 40
    ACO_ITER = 60
    GA_POP = 800
    GA_GEN = 800

    print("Starting Pathfinding Comparison...")

    # 1. Generate Map
    # Returns grid (list of lists) and obstacle_set (set of tuples)
    grid_map, obstacles = generate_map(map_size=MAP_SIZE, seed=MAP_SEED)

    if grid_map is None:
        print("Exiting due to map generation failure.")
        exit()

    # Define goal node based on actual map size
    map_height = len(grid_map)
    map_width = len(grid_map[0]) if map_height > 0 else 0
    goal_node = (map_width - 1, map_height - 1)

    # Verify start/goal are not obstacles (should be handled by safe zones in generator)
    if grid_map[START_NODE[1]][START_NODE[0]] == 0:
        print(f"Error: Start node {START_NODE} is blocked!")
        exit()
    if grid_map[goal_node[1]][goal_node[0]] == 0:
        print(f"Error: Goal node {goal_node} is blocked!")
        exit()

    # If map-only mode is requested, visualize the map and exit
    if args.map_only:
        output_filename = args.output if args.output else f"map_only_{MAP_SIZE}x{MAP_SIZE}_seed{MAP_SEED}.png"
        print(f"Visualizing map only...")
        visualize_map_only(grid_map, obstacles, map_width, START_NODE, goal_node, output_filename)
        exit()

    # Dictionary to store results {Algorithm Name: Path List}
    paths_found = {}
    run_times = {}

    # --- Run Standard A* ---
    print("\nRunning Standard A*...")
    start_time = time.time()
    path_astar_std = astar_standard(grid_map, START_NODE, goal_node)
    run_times['A*'] = time.time() - start_time
    paths_found['A*'] = path_astar_std
    print(f"A* Standard found path length (steps): {len(path_astar_std) - 1 if path_astar_std else 'N/A'}")
    print(f"Time taken: {run_times['A*'] * 1000:.3f} ms")

    # --- Run Standard ACO ---
    print("\nRunning Standard ACO (4-dir)...")
    start_time = time.time()
    path_aco_std = ant_colony_standard(grid_map, START_NODE, goal_node,
                                       n_ants=ACO_ANTS, n_iterations=ACO_ITER)
    run_times['ACO'] = time.time() - start_time
    paths_found['ACO'] = path_aco_std
    print(f"ACO Standard found path length (steps): {len(path_aco_std) - 1 if path_aco_std else 'N/A'}")
    print(f"Time taken: {run_times['ACO'] * 1000:.3f} ms")

    # --- Run Standard GA ---
    print("\nRunning Standard GA (4-dir)...")
    start_time = time.time()
    path_ga_std = genetic_algorithm_standard(grid_map, START_NODE, goal_node,
                                             population_size=GA_POP, generations=GA_GEN)
    run_times['GA'] = time.time() - start_time
    paths_found['GA'] = path_ga_std
    print(f"GA Standard found path length (steps): {len(path_ga_std) - 1 if path_ga_std else 'N/A'}")
    print(f"Time taken: {run_times['GA'] * 1000:.3f} ms")

    # --- Run Hybrid A*-ACO ---
    print("\nRunning Hybrid A*-ACO (8-dir)...")
    start_time = time.time()
    # Instantiate the runner with the generated grid
    hybrid_runner = AStarACOHybridRunner(grid_map, START_NODE, goal_node)
    path_hybrid = hybrid_runner.run()  # This will print internal A* and ACO lengths
    run_times['Hybrid A*-ACO'] = time.time() - start_time
    paths_found['Hybrid A*-ACO'] = path_hybrid
    # Note: Hybrid runner prints its own length calculation during run.
    # We store the path, visualizer will show step count.
    print(f"Hybrid A*-ACO found path length (steps): {len(path_hybrid) - 1 if path_hybrid else 'N/A'}")
    print(f"Time taken: {run_times['Hybrid A*-ACO'] * 1000:.3f} ms")

    # --- Final Summary ---
    print("\n--- Results Summary ---")
    for name, path in paths_found.items():
        steps = len(path) - 1 if path else "Failed"
        # Get runtime in seconds
        runtime_sec = run_times.get(name, -1.0)
        # Convert to milliseconds if valid
        runtime_ms = runtime_sec * 1000 if runtime_sec >= 0 else -1.0

        # Print format in milliseconds
        if runtime_ms >= 0:
            print(f"- {name:<18}: Steps = {steps:<8} Time = {runtime_ms:.3f}ms")
        else:
            print(f"- {name:<18}: Steps = {steps:<8} Time = N/A")

    # 4. Visualize Results
    print("\nGenerating visualization...")
    visualize_paths_single_map(grid_map, obstacles, paths_found, run_times, map_width, START_NODE, goal_node)

    print("\nComparison complete. Close the plot window to exit.")