import heapq
import copy


def heuristic(puzzle):
    h = 0
    expected_list = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2]]
    for i in range(9):
        if puzzle[i] != 0:
            h += abs(expected_list[i][0] - expected_list[puzzle[i]-1][0]) + abs(expected_list[i][1] - expected_list[puzzle[i]-1][1])
    return h


def tile_move(state, start_idx, end_idx):
    state_copy = copy.deepcopy(state)
    curr = state_copy[start_idx]
    state_copy[start_idx] = state_copy[end_idx]
    state_copy[end_idx] = curr
    return state_copy


def succ(state):
    row_col = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
    idx = state.index(0)
    row = row_col[state.index(0)][1]
    col = row_col[state.index(0)][0]
    lst = []

    if row == 0:
        lst.append(tile_move(state, idx, idx+3))
    elif row == 1:
        lst.append(tile_move(state, idx, idx+3))
        lst.append(tile_move(state, idx, idx-3))
    else:
        lst.append(tile_move(state, idx, idx-3))

    if col == 0:
        lst.append(tile_move(state, idx, idx+1))
    elif col == 1:
        lst.append(tile_move(state, idx, idx+1))
        lst.append(tile_move(state, idx, idx-1))
    else:
        lst.append(tile_move(state, idx, idx-1))

    return sorted(lst)


def print_succ(state):
    successful_lst = succ(state)
    for suc in successful_lst:
        print(suc, 'h=' + str(heuristic(suc)))


def solve(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    state_list = []
    open = []
    closed = []
    parent_idx = -1
    parent = {}
    moves = 0
    heapq.heappush(open, (heuristic(state) + parent_idx + 1, state, [parent_idx+1, heuristic(state), parent_idx]))
    while len(open) > 0:
        f, current, info = heapq.heappop(open)
        closed.append(current)
        if current == goal:
            state_list = [current]
            while str(current) in parent.keys():
                current = parent[str(current)]
                state_list.append(current)
            break
        for curr_state in succ(current):
            g_new = info[0] + 1
            f_new = g_new + heuristic(curr_state)
            if curr_state not in closed:
                parent[str(curr_state)] = current
                heapq.heappush(open, (f_new, curr_state, [g_new, heuristic(curr_state), parent_idx + 1]))

    for i in range(len(state_list), 0, -1):
        print(state_list[i-1], 'h=' + str(heuristic(state_list[i-1])), 'moves: ' + str(moves))
        moves += 1