nodes = [(5, 1), (1, 5), (1, 4), (1, 3), (3, 5), (3, 4), (5, 6)]
nodes.sort(key=lambda x: x[0] + x[1])

print(f"node sort: {nodes}")


graph = dict()
count_num = 1

for node in nodes:
    graph[node] = dict(num = count_num)
    print(f"node: {node} value={graph[node]}")
    count_num += 1

def sum(x: int, y: int):
    return x+y

print(f"graph: {graph}")
print(graph[5, 1]["num"])
print(graph[5, 1])
