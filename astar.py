import argparse
import heapq
from itertools import count


BOARD_SIZE = 6


def normalize_board(board):
    board = board.strip().replace(".", "o")
    return board


def parse_board(board):
    board = normalize_board(board)
    raw_pieces = {}
    walls = set()

    for index, cell in enumerate(board):
        row = index // BOARD_SIZE
        col = index % BOARD_SIZE

        if cell == "x":
            walls.add((row, col))
        elif cell != "o":
            raw_pieces.setdefault(cell, []).append((row, col))

    piece_names = sorted(raw_pieces.keys())
    pieces = {}
    start_state = []

    for name in piece_names:
        cells = sorted(raw_pieces[name])
        rows = {row for row, col in cells}
        cols = {col for row, col in cells}

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

    return {
        "piece_names": piece_names,
        "pieces": pieces,
        "walls": walls,
        "start_state": tuple(start_state),
        "a_index": piece_names.index("A"),
    }


def make_grid(info, state):
    grid = [["o" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    for row, col in info["walls"]:
        grid[row][col] = "x"

    for i, name in enumerate(info["piece_names"]):
        piece = info["pieces"][name]
        start = state[i]

        if piece["orientation"] == "H":
            row = piece["fixed"]
            for offset in range(piece["length"]):
                grid[row][start + offset] = name
        else:
            col = piece["fixed"]
            for offset in range(piece["length"]):
                grid[start + offset][col] = name

    return grid


def is_goal(info, state):
    a_piece = info["pieces"]["A"]
    return state[info["a_index"]] + a_piece["length"] == BOARD_SIZE


def heuristic(info, state):
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


def get_neighbors(info, state):
    grid = make_grid(info, state)
    neighbors = []

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
            while col < BOARD_SIZE and grid[row][col] == "o":
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
            while row < BOARD_SIZE and grid[row][col] == "o":
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

            return {
                "line_number": current_line,
                "expected_moves": int(parts[0]),
                "board": normalize_board(parts[1]),
                "cluster_size": int(parts[2]),
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--line",
        type=int,
        default=1,
        help="line number to read from rush.txt",
    )
    args = parser.parse_args()

    puzzle = get_puzzle_from_file("rush.txt", args.line)
    moves = solve_board(puzzle["board"])

    print("line:", puzzle["line_number"])
    print("board:", puzzle["board"])
    print("expected moves:", puzzle["expected_moves"])

    if moves is None:
        print("no solution found")
    else:
        print("found moves:", len(moves))
        print("matches data:", len(moves) == puzzle["expected_moves"])
        print("path:", " ".join(moves))
