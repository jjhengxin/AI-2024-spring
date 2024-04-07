import heapq
# import time

# 计算曼哈顿距离
def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, 3)
                distance += abs(x - i) + abs(y - j)
    return distance

# 获取下一个状态
def get_next_states(state):
    next_states = []
    x, y = next((i, j) for i in range(3) for j in range(3) if state[i][j] == 0)
    for dx, dy, direction in [(0, 1, 'r'), (1, 0, 'd'), (0, -1, 'l'), (-1, 0, 'u')]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            next_state = [row[:] for row in state]
            next_state[x][y], next_state[nx][ny] = next_state[nx][ny], next_state[x][y]
            next_states.append((next_state, direction))
    return next_states

# A*算法求解八数码问题
def solve_puzzle(start, goal):
    pq = [(0 + manhattan_distance(start), 0, start, None)]  # 优先队列：(f值, 代价, 当前状态, 上一步移动方向)
    visited = set()  # 记录访问过的状态
    parent = {}  # 记录每个状态的最优路径上的父状态和移动方向
    g_value = {tuple(map(tuple, start)): 0}  # 记录每个状态的实际代价
    while pq:
        _, cost, current_state, last_move = heapq.heappop(pq)
        if current_state == goal:
            path = []
            while parent.get(tuple(map(tuple, current_state))):
                current_state, move = parent[tuple(map(tuple, current_state))]
                path.append(move)
            return  cost, ''.join(path[::-1])
        if tuple(map(tuple, current_state)) in visited:
            continue
        visited.add(tuple(map(tuple, current_state)))
        for next_state, direction in get_next_states(current_state):
            new_cost = cost + 1
            if new_cost < g_value.get(tuple(map(tuple, next_state)), float('inf')):
                heapq.heappush(pq, (new_cost + manhattan_distance(next_state), new_cost, next_state, direction))
                parent[tuple(map(tuple, next_state))] = (current_state, direction)
                g_value[tuple(map(tuple, next_state))] = new_cost
    return None, None

# 示例
m = input().split()
# start_time = time.perf_counter()
start_state = [[], [], []]
arr = []
for i in range(9):
    if m[i] == 'x':
        start_state[i // 3].append(0)
        x = i // 3
        y = i % 3
    else :
        start_state[i // 3].append(ord(m[i]) - ord('0'))
        arr.append(ord(m[i]) - ord('0'))
# start_state = [[2, 8, 3], [1, 6, 4], [5, 0, 7]]  # 初始状态
        
inversion_count = 0
for i in range(8):
    for j in range(0, i):
        if arr[j]>arr[i]:
            inversion_count += 1

if inversion_count % 2 != 0:
    print('unsolvable')

goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]   # 目标状态

steps, path = solve_puzzle(start_state, goal_state)
if steps is not None:
    # print("Minimum steps to reach the goal state:", steps)
    print(path)
else:
    print("No solution found.")

# end_time = time.perf_counter()
# #print()
# #print(saw)
# print(end_time - start_time)