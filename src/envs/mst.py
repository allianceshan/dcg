import networkx as nx

# 创建一个无向图
G = nx.Graph()

# 添加节点
G.add_node(1, pos=(0, 0))
G.add_node(2, pos=(1, 1))
G.add_node(3, pos=(2, 2))
G.add_node(4, pos=(3, 3))

# 添加带权重的边
G.add_edge(1, 2, weight=2.0)
G.add_edge(1, 3, weight=4.0)
G.add_edge(2, 3, weight=1.0)
G.add_edge(3, 4, weight=3.0)

# 生成最小连通树
T = nx.minimum_spanning_tree(G)

# 打印最小连通树的边及权重
for edge in T.edges(data=True):
    print(edge)