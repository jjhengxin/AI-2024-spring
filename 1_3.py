def heap_init(t):
    while t>1:
        if dist[q[t]] < dist[q[t//2]]:
            q[t], q[t//2] = q[t//2], q[t]
            t = t//2
        else:
            break

def heap_delete(t):
    q[1] = q[t]
    s = 1
    while s*2 <= t:
        l = s*2
        r = s*2+1
        target = s
        if (dist[q[l]] < dist[q[s]]):
            target = l
        if (r <= t and dist[q[r]] < dist[q[target]]):
            target = r
        if (target != s):
            q[s], q[target] = q[target], q[s]
            s = target
        else :
            break
    
def dijkstra():
    q.append(-1)
    lenq = 0
    for x in range(2, n+1):
        q.append(x)
        lenq += 1
        heap_init(lenq)
    lenq = n - 1
    while (lenq >= 1):
        # 找到距起点最近的节点
        new_i = q[1]
        # print(new_i, dist[new_i], "q:", q, dist[6], lenq)
        heap_delete(lenq)
        lenq = lenq - 1
        # 如果是目标节点就直接返回
        if new_i == n :
            if dist[n] == 0x7f7f7f7f : return -1
            else : return dist[n]
        
        # 以该点为中转更新其它节点的距离
        for k in range(1, lenq+1):
            if dist[q[k]] > dist[new_i] + mmap[new_i][q[k]] :
                dist[q[k]] = dist[new_i] + mmap[new_i][q[k]]
                # print("updat:-----------", q[k], dist[q[k]])
                heap_init(k)

    return -1

# 读入数据
n, m = map(int, input().split())
dist = []
mmap = []
for i in range(n+1):
    dist.append(0x7f7f7f7f)
    mmap.append([])
    for j in range(n+1):
        mmap[i].append(0x7f7f7f7f)

dist[1] = 0
for i in range(m):
    a, b, c = map(int, input().split())
    if a == 1 :
        dist[b] = min(dist[b], c)
    else :
        mmap[a][b] = min(mmap[a][b], c)

# 执行函数，打印输出
# print(mmap)
q = [] # 尚未确定最短路的节点
print(dijkstra())