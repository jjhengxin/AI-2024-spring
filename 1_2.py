def dijkstra():
    q = [x for x in range(2,n+1)] # 尚未确定最短路的节点

    while (len(q) > 0):
        new_i = n # n肯定还在队列中
        # 找到距起点最近的节点
        for k in q:
            if dist[k] < dist[new_i]: new_i = k
        # 如果是目标节点就直接返回
        if new_i == n :
            if dist[n] == 0x7f7f7f7f : return -1
            else : return dist[n]
        q.remove(new_i)

        # 以该点为中转更新其它节点的距离
        for k in q:
            if dist[k] > dist[new_i] + mmap[new_i][k] :
                dist[k] = dist[new_i] + mmap[new_i][k]  
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
print(dijkstra())