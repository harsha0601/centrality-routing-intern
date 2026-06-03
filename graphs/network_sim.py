import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Build 50-node weighted graph
np.random.seed(42)
G = nx.erdos_renyi_graph(50, 0.15, seed=42)

# Add latency weights to edges
for u, v in G.edges():
    G[u][v]['latency'] = round(np.random.uniform(1, 10), 2)

# Add positions to nodes
pos = nx.spring_layout(G, seed=42)
for node in G.nodes():
    G.nodes[node]['pos'] = pos[node]

# Print stats
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))

# Generate 3 traffic flows (source → destination pairs)
traffic_flows = [(0, 49), (5, 44), (10, 39)]
for src, dst in traffic_flows:
    try:
        path = nx.shortest_path(G, src, dst, weight='latency')
        print(f"Flow {src}→{dst}: path = {path}")
    except nx.NetworkXNoPath:
        print(f"Flow {src}→{dst}: No path found")

# Draw the network
plt.figure(figsize=(12, 8))
nx.draw_networkx(G, pos, node_size=300, node_color='steelblue',
                 font_size=7, edge_color='gray', alpha=0.7)
plt.title("50-node Network Simulator")
plt.axis('off')
plt.savefig("../results/week2_network.png", dpi=150)
plt.show()
print("Saved to results/week2_network.png")