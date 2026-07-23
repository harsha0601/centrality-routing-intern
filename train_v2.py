"""
Trains GNN-DQN-v2 and DQN-v2 fresh on each of the 10 pre-specified seeds and
writes results/{method}_10seed.csv — no number is ever typed by hand.

Usage:
    python train_v2.py                     # full spec: 2000 episodes, all 10 seeds
    python train_v2.py --episodes 50 --seeds 42          # fast smoke test
    python train_v2.py --episodes 50 --seeds 42,7        # smoke test, 2 seeds

Frozen per the plan (do not change without telling the mentor):
    lr=0.0005, gamma=0.99, buffer=20000, batch=128, episodes=2000,
    epsilon 1.0 -> 0.05 decay 0.998, target net sync every 10 episodes.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from env_v2 import RoutingEnvV2
from models_v2 import GNNDQNPolicyV2, PlainDQNV2, masked_argmax

ALL_SEEDS = [42, 7, 13, 21, 99, 3, 17, 55, 8, 34]

GAMMA = 0.99
LR = 0.0005
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.998
MEM_SIZE = 20_000
BATCH = 128
TARGET_SYNC_EVERY = 10


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, next_mask, done):
        self.buf.append((obs, action, reward, next_obs, next_mask, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, nmask, done = zip(*batch)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(act, dtype=torch.long),
            torch.as_tensor(rew, dtype=torch.float32),
            torch.as_tensor(np.array(nobs), dtype=torch.float32),
            torch.as_tensor(np.array(nmask), dtype=torch.bool),
            torch.as_tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def build_edge_index(env, device):
    edges = []
    for u, v in env.G.edges():
        edges.append([u, v]); edges.append([v, u])
    return torch.as_tensor(edges, dtype=torch.long, device=device).t().contiguous()


def build_batched_edge_index(edge_index, n_nodes, batch_size, device):
    """
    PERFORMANCE FIX: v1's per-sample training loop called the GCN forward pass
    once per item in the minibatch (128 separate 50-node GCNConv calls per
    training step) — this alone was measured at ~3.3s/episode on CPU, i.e.
    ~18+ hours for the full 10-seed x 2000-episode run. That is almost
    certainly why the earlier "real" run never finished in 3-4 hours.

    Fix: since every sample in a minibatch shares the SAME graph topology
    (one seed = one fixed 50-node graph), stack B disconnected copies of the
    graph into one big block-diagonal graph and run GCNConv ONCE per training
    step instead of B times. This is the standard PyG batching trick.
    """
    E = edge_index.shape[1]
    offsets = (torch.arange(batch_size, device=device) * n_nodes).repeat_interleave(E)
    tiled = edge_index.repeat(1, batch_size)
    return tiled + offsets.unsqueeze(0)


def batched_q(model, obs_batch, batched_edge_index, n_nodes, is_gnn):
    """obs_batch: [B, N, F] -> returns q: [B, N]"""
    B = obs_batch.shape[0]
    x_flat = obs_batch.reshape(B * n_nodes, -1)
    if is_gnn:
        q_flat = model(x_flat, batched_edge_index)   # [B*N]
    else:
        q_flat = model(x_flat)                        # [B*N], edge_index unused
    return q_flat.view(B, n_nodes)


def train_one_seed(model_name, seed, episodes, device, verbose=True,
                    collect_diagnostics=False, return_model=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    env = RoutingEnvV2(n_nodes=50, edge_prob=0.15, seed=seed)
    edge_index = build_edge_index(env, device)
    batched_edge_index = build_batched_edge_index(edge_index, env.n_nodes, BATCH, device)
    is_gnn = (model_name == "gnn_dqn_v2")

    ModelCls = GNNDQNPolicyV2 if is_gnn else PlainDQNV2
    policy = ModelCls().to(device)
    target = ModelCls().to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    mem = ReplayBuffer(MEM_SIZE)
    eps = EPS_START

    diag_rewards, diag_losses, diag_maxqs = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_losses, ep_maxqs = 0.0, [], []
        while not done:
            mask = env.get_action_mask()
            if random.random() < eps:
                valid = np.flatnonzero(mask)
                action = int(np.random.choice(valid))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    q = policy(x, edge_index)
                    action = masked_argmax(q, mask, device)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_mask = env.get_action_mask() if not done else mask  # mask irrelevant when done
            mem.push(obs, action, reward, next_obs, next_mask, float(done))
            obs = next_obs
            ep_reward += reward

            if len(mem) >= BATCH:
                b_obs, b_act, b_rew, b_nobs, b_nmask, b_done = (t.to(device) for t in mem.sample(BATCH))

                # --- target Q: Double DQN -- policy net SELECTS the best next
                #     action, target net EVALUATES it. Using the target net for
                #     both (the previous version) systematically overestimates
                #     Q-values, which compounds over training; this is the
                #     standard fix and is applied identically to both models. ---
                with torch.no_grad():
                    qn_select = batched_q(policy, b_nobs, batched_edge_index, env.n_nodes, is_gnn)
                    qn_select = qn_select.masked_fill(~b_nmask, float("-inf"))
                    best_actions = qn_select.argmax(dim=1)

                    qn_eval = batched_q(target, b_nobs, batched_edge_index, env.n_nodes, is_gnn)
                    targets = qn_eval.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    y = b_rew + GAMMA * targets * (1 - b_done)

                # --- current Q for the taken action, batched ---
                qc = batched_q(policy, b_obs, batched_edge_index, env.n_nodes, is_gnn)  # [B, N]
                preds = qc.gather(1, b_act.unsqueeze(1)).squeeze(1)

                loss = F.mse_loss(preds, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                opt.step()
                if collect_diagnostics:
                    ep_losses.append(loss.item())
                    ep_maxqs.append(qc.detach().max().item())

        if ep % TARGET_SYNC_EVERY == 0:
            target.load_state_dict(policy.state_dict())
        eps = max(EPS_END, eps * EPS_DECAY)

        if collect_diagnostics:
            diag_rewards.append(ep_reward)
            diag_losses.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
            diag_maxqs.append(float(np.mean(ep_maxqs)) if ep_maxqs else float("nan"))

        if verbose and ep % max(1, episodes // 5) == 0:
            print(f"    [{model_name} seed={seed}] ep {ep}/{episodes}  eps={eps:.3f}")

    # ---- evaluation: 100 fresh episodes, frozen policy ----
    policy.eval()
    delivered, hops = [], []
    with torch.no_grad():
        for _ in range(100):
            obs, _ = env.reset()
            done, step_count = False, 0
            while not done:
                mask = env.get_action_mask()
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                q = policy(x, edge_index)
                action = masked_argmax(q, mask, device)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step_count += 1
            delivered.append(1 if env.current_node == env.destination else 0)
            hops.append(step_count)

    pdr = 100.0 * np.mean(delivered)
    avg_hops = float(np.mean([h for h, d in zip(hops, delivered) if d == 1])) if any(delivered) else float("nan")
    result = {"seed": seed, "pdr": pdr, "avg_hops": avg_hops, "n_delivered": int(sum(delivered))}

    extras = {}
    if collect_diagnostics:
        extras["diagnostics"] = {"episode_rewards": diag_rewards, "episode_losses": diag_losses, "episode_maxq": diag_maxqs}
    if return_model:
        extras["model"] = policy

    if extras:
        return result, extras
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, ALL_SEEDS)))
    ap.add_argument("--models", type=str, default="gnn_dqn_v2,dqn_v2")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    for model_name in models:
        rows = []
        t0 = time.time()
        for seed in seeds:
            print(f"--- {model_name}, seed {seed} ---")
            row = train_one_seed(model_name, seed, args.episodes, device)
            print(f"    -> PDR={row['pdr']:.2f}%  avg_hops={row['avg_hops']:.2f}")
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.outdir, f"{model_name}_10seed.csv")
        df.to_csv(out_path, index=False)
        elapsed = time.time() - t0
        print(f"Saved {out_path}  ({elapsed/60:.1f} min for {len(seeds)} seed(s))")
        print(f"  mean PDR = {df['pdr'].mean():.2f}% +/- {df['pdr'].std():.2f}")


if __name__ == "__main__":
    main()
