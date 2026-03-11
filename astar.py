import argparse
import heapq
from math import isqrt
from itertools import count


def normalize_board(board):
    board = board.strip().replace(".", "o")
    return board


def parse_board(board):
    board = normalize_board(board)
    size = isqrt(len(board))
    raw_pieces = {}
    walls = set()

    for index, cell in enumerate(board):
        row = index // size
        col = index % size

        if cell == "x":
            walls.add((row, col))
        elif cell != "o":
            raw_pieces.setdefault(cell, []).append((row, col))

    variant = "classic"
    if len(raw_pieces["A"]) == 1:
        variant = "hf4"

    piece_names = sorted(raw_pieces.keys())
    pieces = {}
    start_state = []

    for name in piece_names:
        cells = sorted(raw_pieces[name])
        rows = {row for row, col in cells}
        cols = {col for row, col in cells}

        if variant == "classic":
            if len(rows) == 1:
                orientation = "H"
                fixed = cells[0][0]
                start = min(col for row, col in cells)
            else:
                orientation = "V"
                fixed = cells[0][1]
                start = min(row for row, col in cells)

            pieces[name] = {
                "orientation": orientation,
                "fixed": fixed,
                "length": len(cells),
            }
            start_state.append(start)
        else:
            if len(cells) == 1:
                orientation = "S"
                anchor = cells[0]
            elif len(rows) == 1:
                orientation = "H"
                anchor = (cells[0][0], min(col for row, col in cells))
            else:
                orientation = "V"
                anchor = (min(row for row, col in cells), cells[0][1])

            pieces[name] = {
                "orientation": orientation,
                "length": len(cells),
            }
            start_state.append(anchor)

    return {
        "size": size,
        "variant": variant,
        "piece_names": piece_names,
        "pieces": pieces,
        "walls": walls,
        "start_state": tuple(start_state),
        "a_index": piece_names.index("A"),
        "target": (1, size - 1),
    }


def make_grid(info, state):
    size = info["size"]
    grid = [["o" for _ in range(size)] for _ in range(size)]

    for row, col in info["walls"]:
        grid[row][col] = "x"

    for i, name in enumerate(info["piece_names"]):
        piece = info["pieces"][name]
        if info["variant"] == "classic":
            start = state[i]

            if piece["orientation"] == "H":
                row = piece["fixed"]
                for offset in range(piece["length"]):
                    grid[row][start + offset] = name
            else:
                col = piece["fixed"]
                for offset in range(piece["length"]):
                    grid[start + offset][col] = name
        else:
            row, col = state[i]

            if piece["orientation"] == "S":
                grid[row][col] = name
            elif piece["orientation"] == "H":
                for offset in range(piece["length"]):
                    grid[row][col + offset] = name
            else:
                for offset in range(piece["length"]):
                    grid[row + offset][col] = name

    return grid


def is_goal(info, state):
    if info["variant"] == "classic":
        a_piece = info["pieces"]["A"]
        return state[info["a_index"]] + a_piece["length"] == info["size"]

    return state[info["a_index"]] == info["target"]


def heuristic(info, state):
    if info["variant"] == "hf4":
        row, col = state[info["a_index"]]
        target_row, target_col = info["target"]
        return abs(row - target_row) + abs(col - target_col)

    a_piece = info["pieces"]["A"]
    a_row = a_piece["fixed"]
    a_end = state[info["a_index"]] + a_piece["length"]
    blockers = 0

    for i, name in enumerate(info["piece_names"]):
        if name == "A":
            continue

        piece = info["pieces"][name]
        start = state[i]

        if piece["orientation"] == "H":
            if piece["fixed"] != a_row:
                continue
            end = start + piece["length"] - 1
            if end >= a_end:
                blockers += 1
        else:
            top = start
            bottom = start + piece["length"] - 1
            if top <= a_row <= bottom and piece["fixed"] >= a_end:
                blockers += 1

    return blockers + 1


def get_piece_cells(piece, position):
    if piece["orientation"] == "S":
        row, col = position
        return [(row, col)]

    if piece["orientation"] == "H":
        row, col = position
        return [(row, col + offset) for offset in range(piece["length"])]

    row, col = position
    return [(row + offset, col) for offset in range(piece["length"])]


def get_neighbors(info, state):
    size = info["size"]
    grid = make_grid(info, state)
    neighbors = []

    if info["variant"] == "hf4":
        directions = [(-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")]

        for i, name in enumerate(info["piece_names"]):
            piece = info["pieces"][name]
            row, col = state[i]
            cells = get_piece_cells(piece, (row, col))
            current_cells = set(cells)

            for dr, dc, label in directions:
                can_move = True

                for cell_row, cell_col in cells:
                    next_row = cell_row + dr
                    next_col = cell_col + dc

                    if (
                        next_row < 0
                        or next_row >= size
                        or next_col < 0
                        or next_col >= size
                    ):
                        can_move = False
                        break

                    if (
                        grid[next_row][next_col] != "o"
                        and (next_row, next_col) not in current_cells
                    ):
                        can_move = False
                        break

                if can_move:
                    next_state = list(state)
                    next_state[i] = (row + dr, col + dc)
                    neighbors.append((tuple(next_state), name + label))

        return neighbors

    for i, name in enumerate(info["piece_names"]):
        piece = info["pieces"][name]
        start = state[i]

        if piece["orientation"] == "H":
            row = piece["fixed"]

            distance = 1
            col = start - 1
            while col >= 0 and grid[row][col] == "o":
                next_state = list(state)
                next_state[i] = start - distance
                neighbors.append((tuple(next_state), name + "-" + str(distance)))
                distance += 1
                col -= 1

            distance = 1
            col = start + piece["length"]
            while col < size and grid[row][col] == "o":
                next_state = list(state)
                next_state[i] = start + distance
                neighbors.append((tuple(next_state), name + "+" + str(distance)))
                distance += 1
                col += 1
        else:
            col = piece["fixed"]

            distance = 1
            row = start - 1
            while row >= 0 and grid[row][col] == "o":
                next_state = list(state)
                next_state[i] = start - distance
                neighbors.append((tuple(next_state), name + "-" + str(distance)))
                distance += 1
                row -= 1

            distance = 1
            row = start + piece["length"]
            while row < size and grid[row][col] == "o":
                next_state = list(state)
                next_state[i] = start + distance
                neighbors.append((tuple(next_state), name + "+" + str(distance)))
                distance += 1
                row += 1

    return neighbors


def rebuild_moves(parents, state):
    moves = []

    while state in parents:
        state, move = parents[state]
        moves.append(move)

    moves.reverse()
    return moves


def solve_board(board):
    info = parse_board(board)
    start = info["start_state"]
    best_cost = {start: 0}
    parents = {}
    pq = []
    order = count()

    heapq.heappush(pq, (heuristic(info, start), 0, next(order), start))

    while pq:
        priority, cost, _, state = heapq.heappop(pq)

        if cost != best_cost[state]:
            continue

        if is_goal(info, state):
            return rebuild_moves(parents, state)

        for next_state, move in get_neighbors(info, state):
            next_cost = cost + 1

            if next_state in best_cost and best_cost[next_state] <= next_cost:
                continue

            best_cost[next_state] = next_cost
            parents[next_state] = (state, move)
            next_priority = next_cost + heuristic(info, next_state)
            heapq.heappush(pq, (next_priority, next_cost, next(order), next_state))

    return None


def get_puzzle_from_file(path, line_number):
    with open(path, "r", encoding="utf-8") as file:
        for current_line, line in enumerate(file, start=1):
            if current_line != line_number:
                continue

            parts = line.split()

            if len(parts) == 1:
                return {
                    "line_number": current_line,
                    "expected_moves": None,
                    "board": normalize_board(parts[0]),
                    "cluster_size": None,
                }

            if len(parts) == 2:
                return {
                    "line_number": current_line,
                    "expected_moves": int(parts[0]),
                    "board": normalize_board(parts[1]),
                    "cluster_size": None,
                }

            if len(parts) == 3:
                return {
                    "line_number": current_line,
                    "expected_moves": int(parts[0]),
                    "board": normalize_board(parts[1]),
                    "cluster_size": int(parts[2]),
                }

            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--line",
        type=int,
        default=1,
        help="line number to read from data file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/rush.txt",
        help="path to data file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of consecutive puzzles to solve from rush.txt",
    )
    args = parser.parse_args()

    puzzles = []
    for i in range(args.count):
        puzzles.append(get_puzzle_from_file(args.data, args.line + i))

    for puzzle in puzzles:
        moves = solve_board(puzzle["board"])

        print("line:", puzzle["line_number"])
        print("board:", puzzle["board"])
        print("board size:", isqrt(len(puzzle["board"])))

        if puzzle["expected_moves"] is not None:
            print("expected moves:", puzzle["expected_moves"])

        if moves is None:
            print("no solution found")
        else:
            print("found moves:", len(moves))
            if puzzle["expected_moves"] is not None:
                print("matches data:", len(moves) == puzzle["expected_moves"])
            print("path:", " ".join(moves))
