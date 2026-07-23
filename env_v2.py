"""
RoutingEnv v2 — per the mentor's revision plan, Section 2.

KEY CHANGES from v1:
  - Node features are [deg, bet, clo, load, is_destination] for EVERY node.
    Per the mentor's Section 3 design (Oracle Feature memo): NO distance
    information is given -- is_destination is a plain binary flag. An
    earlier version added dist_to_dest_norm (normalized shortest-path
    distance) to fix a control-blindness issue, but that handed the model
    the answer directly (Dijkstra as a feature), collapsing the GNN-vs-MLP
    comparison. Removed. Proximity to the destination must now be inferred
    via message passing (GNN) or not at all (plain MLP) -- this makes the
    comparison the actual ablation the paper needs.
  - `load` is sampled once per episode, not re-sampled every step.
  - get_action_mask() exposes which next-hop indices are valid neighbours of
    the current node, so invalid actions can be masked to -inf BEFORE argmax,
    instead of being silently replaced by an environment-level fallback.
  - Any n_nodes is supported (50 for training, 65/80/100/150/200 for
    generalisation and stress evals).
  - reset() uses a PERSISTENT episode RNG (self.episode_rng), created ONCE in
    __init__, not recreated on every reset() call.
"""
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces


class RoutingEnvV2(gym.Env):
    def __init__(self, n_nodes=50, edge_prob=0.15, seed=42, max_steps=30):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.seed_val = seed
        self.episode_rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self._build_graph()
        self.action_space = spaces.Discrete(n_nodes)
        # node feature matrix: [n_nodes, 5] -> (deg, bet, clo, load, is_dest)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_nodes, 5), dtype=np.float32)

    def _build_graph(self):
        rng = np.random.RandomState(self.seed_val)
        G = nx.erdos_renyi_graph(self.n_nodes, self.edge_prob, seed=self.seed_val)
        tries = 0
        while not nx.is_connected(G):
            tries += 1
            G = nx.erdos_renyi_graph(self.n_nodes, self.edge_prob, seed=int(rng.randint(1_000_000)))
            if tries > 200:
                raise RuntimeError(
                    f"Could not build a connected graph for n={self.n_nodes}, p={self.edge_prob} "
                    f"after 200 tries — edge_prob is likely below the connectivity threshold for this N."
                )
        for u, v in G.edges():
            G[u][v]["latency"] = round(rng.uniform(1, 5), 2)
        self.G = G
        self.deg = nx.degree_centrality(G)
        self.bet = nx.betweenness_centrality(G)
        self.clo = nx.closeness_centrality(G)
        self.shortest_paths = dict(nx.all_pairs_shortest_path(G))
        self.nodes = list(G.nodes())
        self._centrality = np.array(
            [[self.deg[n], self.bet[n], self.clo[n]] for n in self.nodes], dtype=np.float32
        )
        self._adj = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        for u, v in G.edges():
            self._adj[u, v] = True
            self._adj[v, u] = True

    def _build_obs(self):
        load = self._load  # frozen per-episode, shape [n_nodes]
        is_dest = np.zeros(self.n_nodes, dtype=np.float32)
        is_dest[self.destination] = 1.0
        # Per the mentor's Section 3 design: NO distance information is given.
        # is_dest is a plain binary flag -- proximity to the destination must
        # be inferred (via GCN message passing) or not at all (plain MLP).
        return np.concatenate(
            [self._centrality, load[:, None], is_dest[:, None]], axis=1
        ).astype(np.float32)

    def get_action_mask(self):
        return self._adj[self.current_node].copy()

    def reset(self, seed=None, options=None):
        rng = self.episode_rng if seed is None else np.random.RandomState(seed)
        for _ in range(200):
            self.current_node = int(rng.choice(self.nodes))
            dest_candidates = [n for n in self.nodes if n != self.current_node]
            self.destination = int(rng.choice(dest_candidates))
            path = self.shortest_paths.get(self.current_node, {}).get(self.destination)
            if path is not None and len(path) <= 6:
                break
        self._load = rng.uniform(0, 1, size=self.n_nodes).astype(np.float32)
        self.visited = {self.current_node}
        self.steps = 0
        return self._build_obs(), {}

    def step(self, action):
        self.steps += 1
        mask = self.get_action_mask()
        if not mask[action]:
            raise ValueError(
                f"Action {action} is not a valid neighbour of node {self.current_node}. "
                f"Did you forget to apply get_action_mask() before argmax?"
            )
        delay = self.G[self.current_node][action]["latency"]
        self.visited.add(action)
        prev_node = self.current_node
        self.current_node = action

        if self.current_node == self.destination:
            reward = 20.0 - (self.steps * 0.5)
            terminated = True
        elif self.steps >= self.max_steps:
            reward = -10.0
            terminated = True
        elif action in self.visited and action != self.destination and prev_node != action:
            reward = -2.0
            terminated = False
        else:
            path = self.shortest_paths.get(self.current_node, {}).get(self.destination)
            if path is not None:
                dist = len(path)
                reward = 1.0 / (dist + 1) - delay * 0.05
            else:
                reward = -delay * 0.1
            terminated = False

        return self._build_obs(), reward, terminated, False, {}