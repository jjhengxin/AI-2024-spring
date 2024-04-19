from collections import deque  
"""
待补充代码：对搜索过的格子染色
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def visualize_maze_with_path(maze, path, visited):
    # 创建一个归一化器，将数据值映射到[0, 1]的范围
    norm = BoundaryNorm([0, 0.5, 1.5, 3.5, 5], 4)

    # 定义三种颜色，分别对应数值0、1和2-100的区间
    colors = ['white', 'black', 'lightgreen', 'MediumSeaGreen']

    # 使用 ListedColormap 创建颜色映射
    cmap = ListedColormap(colors)
    # cmap = ListedColormap(['white', 'black', 'LightGreen'])
    if (max(len(maze[0]), len(maze)) >= 8):
        plt.figure(figsize=(0.5*len(maze[0]), 0.5*len(maze)))  # 设置图形大小
    else :
        plt.figure(figsize=(len(maze[0]), len(maze)))
    #plt.imshow(maze, cmap='Greys', interpolation='nearest')  # 使用灰度色图，并关闭插值
    #plt.imshow(maze, cmap=cmap, interpolation='nearest')  # 使用自定义颜色映射，并关闭插值
    plt.imshow(maze, cmap=cmap, norm=norm)
    plt.colorbar()  # 添加颜色条


    # 设置坐标轴刻度和边框
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=3)
    
    plt.axis('on')  # 显示坐标轴
    # # 绘制路径
    # if path:
    #     path_x, path_y = zip(*path)
    #     plt.plot(path_y, path_x, marker='o', markersize=8, color='HotPink', linewidth=3)
    #     path_x, path_y = zip(*visited)
    #     plt.scatter(path_y, path_x, marker='o',color='SlateGray')
    for i in range(len(visited)):
        
        # 绘制访问过的点
        if visited:
            visited_x, visited_y = zip(*visited[:i+1])
            plt.scatter(visited_y, visited_x, marker='s', color='SlateGray', alpha=0.5, s=200)
        
        
        if (max(len(maze[0]), len(maze)) >= 8):
            plt.pause(0.1)  # 暂停0.5秒，显示当前点
        else :
            plt.pause(0.5) 
    if path:
         path_x, path_y = zip(*path)
         plt.plot(path_y, path_x, marker='o', markersize=15, color='Khaki', linewidth=3)
    plt.text(len(maze[0])/3, -0.8, "最短距离："+str(dis[n-1][m-1]), fontsize=12, fontproperties='SimHei',bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
 
def bfs(maze, start, end):  
    rows, cols = len(maze), len(maze[0])  
    print("rows: ", rows, "cols: ", cols)
    visited = [[False for _ in range(cols)] for _ in range(rows)]    
    prev = [[[-1, -1] for _ in range(cols)] for _ in range(rows)]
    # print(visited[14][19])  
    # print(prev)
    queue = deque([(start[0], start[1])])  
    visited[start[0]][start[1]] = True 
    vv.append((start[0], start[1])) 
  
    while queue:  
        x, y = queue.popleft()  
        # print(x, y)
        
  
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 右、左、下、上  
            nx, ny = x + dx, y + dy 
            print(nx, ny)
            # print(nx , ny, visited[nx][ny], maze[nx][ny]) 
            if 0 <= nx < rows and 0 <= ny < cols :
                if visited[nx][ny] or maze[nx][ny] == 1:  
                    continue
                queue.append((nx, ny))  
                
                visited[nx][ny] = True  
                vv.append((nx, ny))
                dis[nx][ny] = dis[x][y] + 1
                # print(nx, ny, prev)
                prev[nx][ny] = [x, y]  
                if nx == end[0] and ny == end[1]: 
                    
                    print(visited)
                    # vv = []
                    # for i in range(n):
                    #     for j in range(m):
                    #         if (visited[i][j]):vv.append((i,j))
                    # print(vv)
                    print(dis[n-1][m-1])
                    return reconstruct_path(prev, end)  
  
    return None 
 
    
def dijkstra(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    prev = [[[-1, -1] for _ in range(cols)] for _ in range(rows)]
    global dis
    dis = [[float('inf') for _ in range(cols)] for _ in range(rows)]  # 初始化距离为无穷大
    dis[start[0]][start[1]] = 0  # 起始点距离设为0
    vv.append((start[0], start[1])) 

    queue = [(0, start[0], start[1])]  # 使用元组 (距离, x坐标, y坐标) 存储节点信息

    while queue:
        dist, x, y = heapq.heappop(queue)  # 从优先队列中取出距离最小的节点
        if visited[x][y]:
            continue
        visited[x][y] = True

        if x == end[0] and y == end[1]:
            return reconstruct_path(prev, end)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 右、左、下、上
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and maze[nx][ny] != 1:
                new_dist = dist + maze[nx][ny] + 1   # 更新到达邻居节点的距离
                if (nx, ny) not in vv : vv.append((nx, ny))
                if new_dist < dis[nx][ny]:  # 如果更新后的距离小于原有距离，则更新距离并加入优先队列
                    dis[nx][ny] = new_dist
                    heapq.heappush(queue, (new_dist, nx, ny))
                    prev[nx][ny] = [x, y]  # 更新前驱节点

    return None

def dfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    prev = [[[-1, -1] for _ in range(cols)] for _ in range(rows)]
    stack = [(start[0], start[1])]
    visited[start[0]][start[1]] = True
    vv.append((start[0], start[1])) 
    is_find = 0
    res = rows * cols + 1
    while stack:
        x, y = stack.pop()  # 使用栈来实现深度优先搜索
        vv.append((x, y))
        if x == end[0] and y == end[1]:
            is_find = 1
            res = min(res, dis[n-1][m-1])
            print(dis[n-1][m-1])


        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 右、左、下、上
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (not visited[nx][ny] or dis[x][y]+1 < dis[nx][ny]) and maze[nx][ny] == 0:
                stack.append((nx, ny))
                visited[nx][ny] = True
                
                dis[nx][ny] = dis[x][y] + 1
                prev[nx][ny] = [x, y]

    if is_find == 1 :return reconstruct_path(prev, end)
    else : return None 


import heapq

def heuristic_cost_estimate(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])  # 曼哈顿距离

def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    prev = [[[-1, -1] for _ in range(cols)] for _ in range(rows)]
    g_value = [[float('inf') for _ in range(cols)] for _ in range(rows)]
    g_value[start[0]][start[1]] = 0
    pq = [(0 + heuristic_cost_estimate(start, end), start[0], start[1])]  # 优先队列：(f值, x坐标, y坐标)

    while pq:
        _, x, y = heapq.heappop(pq)
        if x == end[0] and y == end[1]:
            vv.append((x, y))
            print(g_value[end[0]][end[1]])
            global dis
            dis = g_value
            return reconstruct_path(prev, end)

        if visited[x][y]:
            continue
        visited[x][y] = True
        vv.append((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 右、左、下、上
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != 1:
                tentative_g_value = g_value[x][y] + maze[nx][ny] + 1
                if tentative_g_value < g_value[nx][ny]:
                    g_value[nx][ny] = tentative_g_value
                    heapq.heappush(pq, (tentative_g_value + heuristic_cost_estimate((nx, ny), end), nx, ny))
                    prev[nx][ny] = [x, y]

    return None



def reconstruct_path(prev, end):  
    print("find the path----------------")
    path = []  
    while end != [-1, -1]:  
        path.append(tuple(end))  
        end = prev[end[0]][end[1]]  
    return path[::-1]  # 反转路径，使其从起点到终点  
  
n, m = map(int, input().split())

maze = []
dis = []
vv = []
for i in range(n):
    maze.append([])
    dis.append([])
    maze[i] = list(map(int, input().split()))
    for j in range(m): dis[i].append(-1)

dis[0][0] = 0
# print(maze)
start = (0, 0)
end = (n-1, m-1)

path = dfs(maze, start, end)  
if path:  
    print("result path:    ",path)
   
print("explore process:  ",vv)

path = tuple(map(tuple, path))
print(dis[n-1][m-1])
# print(path)
# 可视化迷宫及路径
visualize_maze_with_path(maze, path, vv)
