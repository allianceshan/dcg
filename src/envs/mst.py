import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_grid_graph(grid_size):
    G = nx.grid_2d_graph(grid_size, grid_size)
    return G

def get_min_spanning_tree(G):
    return nx.minimum_spanning_tree(G)

def move_agents(agents, min_spanning_tree):
    for agent_id in range(len(agents)):
        agent = agents[agent_id]
        neighbors = list(min_spanning_tree.neighbors(agent))
        if neighbors:
            next_move = random.choice(neighbors)
            agents[agent_id] = next_move

def plot_agents_and_tree(min_spanning_tree, agents):
    pos = {(x, y): (y, -x) for x, y in min_spanning_tree.nodes()}
    nx.draw(min_spanning_tree, pos, with_labels=False, node_size=10, node_color='b', font_size=8)
    for agent_id, agent in enumerate(agents):
        plt.scatter(agent[1], -agent[0], label=f'Agent {agent_id + 1}', s=50, c='r')
    plt.title("Agents' Positions and Minimum Spanning Tree")

grid_size = 10
n_agents = 3
n_steps = 10

G = generate_grid_graph(grid_size)
G.add_weight_eage_from()
min_spanning_tree = get_min_spanning_tree(G)
agents = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(n_agents)]
plot_agents_and_tree(min_spanning_tree, agents)
plt.legend()


for step in range(n_steps):
    move_agents(agents, min_spanning_tree)
    G = generate_grid_graph(grid_size)  # 重新生成网格图
    min_spanning_tree = get_min_spanning_tree(G)

    plot_agents_and_tree(min_spanning_tree, agents)
    plt.legend()
    plt.show()
