def bfs():
    que = [1]
    s = 0
    e = 0
    while s<=e:
        for k in mmap[que[s]]:
            if dis[k] == -1:
                dis[k] = dis[que[s]] + 1
                e += 1
                que.append(k)
            if k == n:return dis[n]
        s += 1        
    return -1
    


n, m = map(int, input().split())

mmap = []
dis = []
for i in range(n+1):
    mmap.append([])
    dis.append(-1)
dis[1] = 0

for i in range(m):
    a, b = map(int, input().split())
    mmap[a].append(b)

print(bfs())


