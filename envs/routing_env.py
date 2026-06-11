import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class RoutingEnv(gym.Env):
    def __init__(self, n_nodes=50, edge_prob=0.15, seed=42):
        super().__init__()
        self.n_nodes = n_nodes
        self.seed_val = seed
        self.G = None
        self.current_node = None
        self.destination = None
        self.visited = None
        self.steps = 0
        self.max_steps = 50

        # Action = which node to go to next (0 to n_nodes-1)
        self.action_space = spaces.Discrete(n_nodes)

        # State = [degree, betweenness, closeness, traffic_load,
        #          dst_degree, dst_betweenness, dst_closeness,
        #          steps_remaining]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float32)

        self._build_graph()

    def _build_graph(self):
        np.random.seed(self.seed_val)
        self.G = nx.erdos_renyi_graph(
            self.n_nodes, 0.15, seed=self.seed_val)
        for u, v in self.G.edges():
            self.G[u][v]['latency'] = round(
                np.random.uniform(1, 10), 2)
        self.deg = nx.degree_centrality(self.G)
        self.bet = nx.betweenness_centrality(self.G)
        self.clo = nx.closeness_centrality(self.G)

    def _get_obs(self):
        return np.array([
            self.deg[self.current_node],
            self.bet[self.current_node],
            self.clo[self.current_node],
            np.random.uniform(0, 1),  # traffic load
            self.deg[self.destination],
            self.bet[self.destination],
            self.clo[self.destination],
            (self.max_steps - self.steps) / self.max_steps
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        nodes = list(self.G.nodes())
        self.current_node = np.random.choice(nodes)
        self.destination = np.random.choice(
            [n for n in nodes if n != self.current_node])
        self.visited = set([self.current_node])
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        neighbors = list(self.G.neighbors(self.current_node))

        # If action not a valid neighbor, pick best neighbor
        if action not in neighbors:
            action = max(neighbors, key=lambda n: self.bet[n])

        delay = self.G[self.current_node][action]['latency']
        self.visited.add(action)
        self.current_node = action

        # Reward
        if self.current_node == self.destination:
            reward = 10.0 - (delay * 0.1)  # delivered + low delay bonus
            terminated = True
        elif self.steps >= self.max_steps:
            reward = -5.0  # timeout penalty
            terminated = True
        elif action in self.visited:
            reward = -1.0  # loop penalty
            terminated = False
        else:
            reward = -delay * 0.1  # small delay penalty per hop
            terminated = False

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        print(f"Step {self.steps}: "
              f"Node {self.current_node} → Dest {self.destination}")