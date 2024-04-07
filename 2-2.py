from collections import deque   
import time

def bfs(initial, target):
    queue = deque([(initial, 0)])     #双端队列
    enqueued = set([tuple(initial)])

    while queue:
        node, depth = queue.popleft()
        
        i = node.index('x')

        # 空格的行号和列号
        x, y = i // 3, i % 3

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for x1, y1 in moves:
            x2, y2 = x + x1, y + y1
            # 检查新的位置是否在九宫格内
            if 0 <= x2 < 3 and 0 <= y2 < 3:
                node1 = node.copy()
                node1[i], node1[x2 * 3 + y2] = node1[x2 * 3 + y2], node1[i]
                if(node1 == target):
                    return depth+1
                
                node2 = tuple(node1)
                if node2 not in enqueued:
                    enqueued.add(node2)
                    queue.append((node1, depth + 1))
            
    return -1


initial = list((input().split()))


# 记录程序开始时间
start_time = time.perf_counter()

for i in range(0,9):
    if(initial[i] != 'x'):
        initial[i] = int(initial[i])
    
target = [1,2,3,4,5,6,7,8,'x']

if(initial == target):
    print(0)
    exit(0)

print(bfs(initial,target))

# # 记录程序结束时间
# end_time = time.perf_counter()

# # 计算并打印运行时间
# running_time = end_time - start_time
# print(f'程序运行时间: {running_time}秒')