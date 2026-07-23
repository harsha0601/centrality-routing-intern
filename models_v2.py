"""
Model A: GNN-DQN-v2 (proposed) and Model B: DQN-v2 control, per the mentor's
revision plan Section 2. The ONLY architectural difference between A and B is
the presence of the GCN encoder — everything else (masking, head shape family,
hyperparameters) is matched, which is the whole point of the comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

NODE_FEATURES = 5  # deg, bet, clo, load, is_destination (binary flag, no distance oracle -- per mentor's Section 3 design)


class GNNDQNPolicyV2(nn.Module):
    """Model A — GCN encoder + per-node Q-value head."""

    def __init__(self, node_features=NODE_FEATURES, hidden_dim=32, embedding_dim=16):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.head1 = nn.Linear(embedding_dim, 64)
        self.head2 = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        """
        x: [N, node_features] node feature matrix for THIS state (destination
           flag and load already baked in by the caller — see env_v2._build_obs)
        edge_index: [2, E] static graph connectivity
        returns: q [N] — one Q-value per node (no pooling, no fixed-size head)
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))     # [N, embedding_dim] — per-node, never pooled
        q = F.relu(self.head1(h))
        q = self.head2(q).squeeze(-1)              # [N]
        return q


class PlainDQNV2(nn.Module):
    """Model B — matched per-node MLP head, no GCNConv layers at all."""

    def __init__(self, node_features=NODE_FEATURES, hidden_dim=64):
        super().__init__()
        self.head1 = nn.Linear(node_features, hidden_dim)
        self.head2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index=None):
        # edge_index accepted (and ignored) so both models share one call signature
        q = F.relu(self.head1(x))
        q = self.head2(q).squeeze(-1)
        return q


def masked_argmax(q, mask, device):
    """q: [N] tensor of Q-values. mask: [N] bool array, True = valid neighbour."""
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    masked_q = q.masked_fill(~mask_t, float("-inf"))
    return int(masked_q.argmax().item())


def masked_max(q, mask, device):
    """Same as above but returns the max VALUE, for target-Q computation.
    This is the step the plan explicitly warns is easy to forget."""
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    masked_q = q.masked_fill(~mask_t, float("-inf"))
    return masked_q.max()
