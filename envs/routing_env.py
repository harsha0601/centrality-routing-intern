import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class RoutingEnv(gym.Env):
    def __init__(self, n_nodes=50, edge_prob=0.15, seed=42):
        super().__init__()
        self.n_nodes = n_nodes
        self.seed_val = seed
        self.max_steps = 30  # reduced from 50
        self.G = None
        self._build_graph()

        self.action_space = spaces.Discrete(n_nodes)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float32)

    def _build_graph(self):
        np.random.seed(self.seed_val)
        self.G = nx.erdos_renyi_graph(
            self.n_nodes, 0.15, seed=self.seed_val)
        # Ensure connected
        while not nx.is_connected(self.G):
            self.G = nx.erdos_renyi_graph(
                self.n_nodes, 0.15, seed=np.random.randint(1000))
        for u, v in self.G.edges():
            self.G[u][v]['latency'] = round(
                np.random.uniform(1, 5), 2)  # reduced max latency
        self.deg = nx.degree_centrality(self.G)
        self.bet = nx.betweenness_centrality(self.G)
        self.clo = nx.closeness_centrality(self.G)
        # Pre-compute shortest paths
        self.shortest_paths = dict(nx.all_pairs_shortest_path(self.G))

    def _get_obs(self):
        return np.array([
            self.deg[self.current_node],
            self.bet[self.current_node],
            self.clo[self.current_node],
            np.random.uniform(0, 1),
            self.deg[self.destination],
            self.bet[self.destination],
            self.clo[self.destination],
            (self.max_steps - self.steps) / self.max_steps
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        nodes = list(self.G.nodes())
        # Pick src/dst that are reachable and close (max 5 hops)
        for _ in range(100):
            self.current_node = np.random.choice(nodes)
            self.destination  = np.random.choice(
                [n for n in nodes if n != self.current_node])
            try:
                path = self.shortest_paths[self.current_node][self.destination]
                if len(path) <= 6:  # only pick easy routes
                    break
            except:
                continue
        self.visited = set([self.current_node])
        self.steps = 0
        self.optimal_path = self.shortest_paths.get(
            self.current_node, {}).get(self.destination, [])
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        neighbors = list(self.G.neighbors(self.current_node))

        # If invalid action, pick best neighbor toward destination
        if action not in neighbors:
            # Guide toward destination using shortest path
            if (self.current_node in self.shortest_paths and
                self.destination in self.shortest_paths[self.current_node]):
                path = self.shortest_paths[self.current_node][self.destination]
                action = path[1] if len(path) > 1 else neighbors[0]
            else:
                action = neighbors[0]

        delay = self.G[self.current_node][action]['latency']
        self.visited.add(action)
        self.current_node = action

        # Richer reward shaping
        if self.current_node == self.destination:
            reward = 20.0 - (self.steps * 0.5)  # bonus for fewer hops
            terminated = True
        elif self.steps >= self.max_steps:
            reward = -10.0
            terminated = True
        elif action in self.visited and action != self.destination:
            reward = -2.0  # loop penalty
            terminated = False
        else:
            # Progress reward — closer to destination = higher reward
            if (self.current_node in self.shortest_paths and
                self.destination in self.shortest_paths[self.current_node]):
                dist = len(self.shortest_paths[self.current_node][self.destination])
                reward = 1.0 / (dist + 1) - delay * 0.05
            else:
                reward = -delay * 0.1
            terminated = False

        return self._get_obs(), reward, terminated, False, {}