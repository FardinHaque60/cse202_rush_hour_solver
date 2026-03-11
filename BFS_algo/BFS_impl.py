import time
import csv
import tracemalloc
import multiprocessing
from collections import deque


def load_puzzles(filepath: str, size: int):
    """Loads puzzles from the text file based on the grid size format."""
    puzzles = []
    moves_list = []
    with open(filepath, "r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            if size == 6:
                parts = stripped.split(maxsplit=2)
                if len(parts) != 3:
                    continue
                moves, board_desc, _ = parts
            else:  # 4x4
                parts = stripped.split(" ")
                if len(parts) != 2:
                    continue
                moves, board_desc = parts

            puzzles.append(board_desc)
            moves_list.append(int(moves))
    return puzzles, moves_list


def generate_next_states(grid_str: str) -> list[str]:
    """Returns all valid possible moves for the input grid state."""
    n = int(len(grid_str) ** 0.5)
    if n * n != len(grid_str):
        raise ValueError("grid_str length must be a square")

    grid = [list(grid_str[i * n:(i + 1) * n]) for i in range(n)]

    # Occupied cells for each piece (A-Z, excluding 'o')
    pieces = {}
    for r in range(n):
        for c in range(n):
            ch = grid[r][c]
            if ch != "o":
                pieces.setdefault(ch, []).append((r, c))

    def orientation(cells):
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) == 1:
            return "single"
        if len(rows) == 1:
            return "h"  # horizontal
        if len(cols) == 1:
            return "v"  # vertical
        raise ValueError(f"invalid piece shape for cells: {cells}")

    def can_move(cells_set, dr, dc, step):
        for r, c in cells_set:
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < n and 0 <= nc < n):
                return False
            if grid[nr][nc] != "o" and (nr, nc) not in cells_set:
                return False
        return True

    def apply_move(piece, cells, dr, dc, step):
        new_grid = [row[:] for row in grid]
        for r, c in cells:
            new_grid[r][c] = "o"
        for r, c in cells:
            nr, nc = r + dr * step, c + dc * step
            new_grid[nr][nc] = piece
        return "".join("".join(row) for row in new_grid)

    results = []
    seen = set()

    for piece, cells in pieces.items():
        cells_set = set(cells)
        orient = orientation(cells)

        directions = []
        if orient in ("h", "single"):
            directions.extend([(0, -1), (0, 1)])
        if orient in ("v", "single"):
            directions.extend([(-1, 0), (1, 0)])

        for dr, dc in directions:
            step = 1
            while can_move(cells_set, dr, dc, step):
                state = apply_move(piece, cells, dr, dc, step)
                if state not in seen:
                    seen.add(state)
                    results.append(state)
                step += 1

    return results


def bfs_path_to_target(grid_str: str):
    """
    Finds the shortest path from grid_str to a target state using BFS.
    Returns the path and the number of states visited.
    """
    n = int(len(grid_str) ** 0.5)

    def check_target_condition(state: str):
        """
        Dynamically checks if the 'A' car has reached the edge or has a clear path.
        Returns (is_target: bool, needs_extra_move_to_exit: bool)
        """
        rightmost_a = state.rfind('A')
        if rightmost_a == -1:
            return False, False

        c = rightmost_a % n
        r = rightmost_a // n

        # If 'A' has physically reached the rightmost edge
        if c == n - 1:
            return True, False

        # Check if the path is clear to the right
        for i in range(c + 1, n):
            if state[r * n + i] != 'o':
                return False, False

        # Path is clear, but 'A' hasn't physically moved to the exit yet
        return True, True

    q = deque([grid_str])
    parent = {grid_str: None}

    while q:
        cur = q.popleft()

        is_target, needs_extra_move = check_target_condition(cur)

        if is_target:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]

            # Convert nodes to edges (moves). Add 1 if the path is clear but car hasn't exited.
            moves_taken = (len(path) - 1) + (1 if needs_extra_move else 0)
            return path[::-1], moves_taken, len(parent)

        for nxt in generate_next_states(cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    return None, -1, len(parent)


def solve_board_task(args):
    """Worker function for multiprocessing pool."""
    idx, board, expected_moves = args

    # Start tracking memory strictly for this worker process
    tracemalloc.start()
    start_time = time.perf_counter()

    sol_path, moves_taken, visited_count = bfs_path_to_target(board)

    runtime = time.perf_counter() - start_time

    # Get peak memory used by this task, then stop tracking
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    if sol_path is None:
        return idx, board, -1, expected_moves, visited_count, runtime, peak_mem_mb, "FAILED (No Solution)"

    status = "OK" if moves_taken == expected_moves else f"MISMATCH (Got {moves_taken}, Expected {expected_moves})"

    return idx, board, moves_taken, expected_moves, visited_count, runtime, peak_mem_mb, status


def main():
    # Change the filepath and size to test different board sizes
    filepath = "../data/rush_no_walls.txt"
    grid_size = 6
    output_csv = "bfs_results.csv"

    try:
        puzzles, expected = load_puzzles(filepath, grid_size)
    except FileNotFoundError:
        print(f"Could not find data file: {filepath}")
        return

    print(f"Loaded {len(puzzles)} puzzles from {filepath}.")

    # We create a list of tuples containing the arguments for our worker function
    tasks = [(i, puzzles[i], expected[i]) for i in range(len(puzzles))]

    cpu_cores = multiprocessing.cpu_count()
    print(f"Starting multiprocessing pool with {cpu_cores} workers...")
    print(f"Writing results to {output_csv}...\n")

    start_time = time.time()
    total_mem = 0.0
    # Open CSV for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the CSV header
        writer.writerow(
            ["Index", "Board", "Moves", "Expected_Moves", "Visited_States", "Runtime_sec", "Peak_Memory_MB", "Status"])

        # Run the BFS searches in parallel
        with multiprocessing.Pool(processes=cpu_cores) as pool:
            for res in pool.imap_unordered(solve_board_task, tasks):
                idx, board, moves, exp, visited, rt, mem, status = res
                total_mem += mem
                # Save to CSV
                writer.writerow([idx, board, moves, exp, visited, rt, mem, status])

                print(
                    f"Board {idx:04d} | Moves: {moves:2d} (Exp: {exp:2d}) | Visited: {visited:6d} | Time: {rt:.4f}s | Mem: {mem:.2f} MB | {status}")

    total_time = time.time() - start_time
    print(f"\nFinished processing {len(tasks)} boards in {total_time:.2f} seconds.")
    print(f"Total mem: {total_mem:.2f} MB")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()