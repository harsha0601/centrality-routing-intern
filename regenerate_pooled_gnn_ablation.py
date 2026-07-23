"""
Regenerates row 13 of the provenance table honestly. This does NOT fix the
state bug (that's what v2 is for) -- it reproduces exactly what the original
notebooks/statistical_tests.ipynb cell 1 methodology actually computes when
you let it run for real, and saves it to a CSV so it becomes traceable
instead of either fabricated or missing.

KNOWN LIMITATION, report alongside these numbers: the loaded policy's forward
pass never receives current-node/destination (see gnn_dqn_train.ipynb), so
its action is ~constant per seed and PDR here is largely driven by
RoutingEnv's fallback-to-shortest-path substitution for invalid actions, not
genuine routing decisions. This number is REAL (traceable, reproducible) but
still describes a broken/uninformative policy -- exactly the finding your
mentor's memo anticipated.
"""
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

from envs.routing_env import RoutingEnv


class GNNDQNPolicy(nn.Module):
    """Exact architecture from gnn_dqn_train.ipynb / evaluation.ipynb --
    must match to load gnn_dqn_weights.pt."""
    def __init__(self, node_features=4, hidden_dim=32, embedding_dim=16, n_actions=50):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long)
        graph_embed = global_mean_pool(h, batch)
        q = F.relu(self.fc1(graph_embed))
        return self.fc2(q)


SEEDS_10 = [42, 7, 13, 21, 99, 3, 17, 55, 8, 34]


def main():
    policy = GNNDQNPolicy()
    policy.load_state_dict(torch.load("results/gnn_dqn_weights.pt", weights_only=False))
    policy.eval()

    rows = []
    for seed in SEEDS_10:
        np.random.seed(seed)
        G = nx.erdos_renyi_graph(50, 0.15, seed=seed)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(50, 0.15, seed=np.random.randint(1000))
        for u, v in G.edges():
            G[u][v]["latency"] = round(np.random.uniform(1, 5), 2)

        deg = nx.degree_centrality(G)
        bet = nx.betweenness_centrality(G)
        clo = nx.closeness_centrality(G)

        node_features = []
        for node in G.nodes():
            node_features.append([deg[node], bet[node], clo[node], np.random.uniform(0, 1)])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_list = []
        for u, v in G.edges():
            edge_list.append([u, v]); edge_list.append([v, u])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        env = RoutingEnv(n_nodes=50, edge_prob=0.15, seed=seed)

        # The model's forward pass depends only on (x, edge_index), which are
        # fixed for this seed's graph -- it never sees current-node/destination
        # (that's the state bug). So the action is provably identical on every
        # step; compute it once instead of thousands of redundant times.
        with torch.no_grad():
            q_vals = policy(x, edge_index)
            fixed_action = q_vals[0].argmax().item()
        print(f"  seed {seed}: fixed action = node {fixed_action} (chosen identically on every step)")

        delivered, hop_list = [], []
        for _ in range(100):
            obs, _ = env.reset()
            done, total_reward, hops = False, 0.0, 0
            while not done:
                action = fixed_action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                hops += 1
                # Defensive cap: envs/routing_env.py's step() has a real bug --
                # its "already visited" early-return path never checks
                # self.steps >= self.max_steps, so a fixed action that keeps
                # landing on a visited node truncates only through this guard.
                if hops >= env.max_steps:
                    total_reward = -10.0  # treat as a timeout, matching the env's own convention
                    break
            delivered.append(1 if total_reward > 5 else 0)
            hop_list.append(hops)

        pdr = 100.0 * np.mean(delivered)
        avg_hops = float(np.mean([h for h, d in zip(hop_list, delivered) if d == 1])) if any(delivered) else float("nan")
        n_delivered = int(sum(delivered))
        print(f"seed {seed}: PDR={pdr:.2f}%  avg_hops={avg_hops:.2f}  n_delivered={n_delivered}/100")
        rows.append({"seed": seed, "pdr": pdr, "avg_hops": avg_hops, "n_delivered": n_delivered})

    df = pd.DataFrame(rows)
    df.to_csv("results/gnn_dqn_pooled_regenerated_10seed.csv", index=False)
    print(f"\nSaved results/gnn_dqn_pooled_regenerated_10seed.csv")
    print(f"mean PDR = {df['pdr'].mean():.2f}% +/- {df['pdr'].std():.2f}")
    print("\nCompare to the manuscript's claimed 70.60% +/- 11.33% and to the")
    print("hand-typed gnn_pdrs list found in ablation_dqn_no_gcn.ipynb:")
    print("[90.0, 84.0, 79.0, 75.0, 71.0, 69.0, 66.0, 61.0, 56.0, 55.0]")


if __name__ == "__main__":
    main()