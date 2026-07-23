"""
Section 5 stress evaluations, pre-registered in the mentor's
"DQN-v2 Instability" memo -- run to restore discriminative power now that
PDR is saturated at 100% for both models on the training topology and the
original 3 generalisation graphs.

PRE-REGISTERED CONFIGS (do not add/drop/swap after seeing results):
  Sparse unseen graphs, near connectivity threshold:
    N=50  p=0.08 (seed 21)
    N=80  p=0.06 (seed 34)
    N=100 p=0.05 (seed 55)
  Larger unseen graphs:
    N=150 p=0.05 (seed 3)
    N=200 p=0.04 (seed 17)
  Tight step budget:
    the original 3 unseen graphs (80/100/65) with max_steps halved (15 instead of 30)
  Random-policy baseline on every config above.

IMPORTANT: saves incrementally, after EVERY seed, not just once at the end.
If this run is interrupted (e.g. Colab disconnect), just rerun this same
script -- it automatically skips any (model, seed) pair already present in
results/stress_eval.csv and only computes what's missing.

Usage: python stress_eval.py
Output: results/stress_eval.csv, results/stress_eval_random_baseline.csv
"""
import os
import numpy as np
import pandas as pd
import torch

from env_v2 import RoutingEnvV2
from models_v2 import masked_argmax
from train_v2 import train_one_seed, build_edge_index, ALL_SEEDS

TRAIN_EPISODES = 3000
CSV_PATH = "results/stress_eval.csv"
RANDOM_CSV_PATH = "results/stress_eval_random_baseline.csv"

SPARSE_CONFIGS = [
    {"label": "sparse_N50_p0.08",  "n_nodes": 50,  "edge_prob": 0.08, "graph_seed": 21},
    {"label": "sparse_N80_p0.06",  "n_nodes": 80,  "edge_prob": 0.06, "graph_seed": 34},
    {"label": "sparse_N100_p0.05", "n_nodes": 100, "edge_prob": 0.05, "graph_seed": 55},
]
LARGE_CONFIGS = [
    {"label": "large_N150_p0.05", "n_nodes": 150, "edge_prob": 0.05, "graph_seed": 3},
    {"label": "large_N200_p0.04", "n_nodes": 200, "edge_prob": 0.04, "graph_seed": 17},
]
TIGHT_BUDGET_CONFIGS = [
    {"label": "tightbudget_N80",  "n_nodes": 80,  "edge_prob": 0.12, "graph_seed": 99},
    {"label": "tightbudget_N100", "n_nodes": 100, "edge_prob": 0.10, "graph_seed": 7},
    {"label": "tightbudget_N65",  "n_nodes": 65,  "edge_prob": 0.13, "graph_seed": 13},
]
HALVED_MAX_STEPS = 15
ALL_CONFIGS = SPARSE_CONFIGS + LARGE_CONFIGS


def eval_policy_on_graph(policy, device, n_nodes, edge_prob, graph_seed, max_steps=30, n_eval=100):
    env = RoutingEnvV2(n_nodes=n_nodes, edge_prob=edge_prob, seed=graph_seed, max_steps=max_steps)
    edge_index = build_edge_index(env, device)
    policy.eval()
    delivered, hops = [], []
    with torch.no_grad():
        for _ in range(n_eval):
            obs, _ = env.reset()
            done, steps = False, 0
            while not done:
                mask = env.get_action_mask()
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                q = policy(x, edge_index)
                action = masked_argmax(q, mask, device)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            delivered.append(1 if env.current_node == env.destination else 0)
            hops.append(steps)
    pdr = 100.0 * np.mean(delivered)
    avg_hops = float(np.mean([h for h, d in zip(hops, delivered) if d == 1])) if any(delivered) else float("nan")
    return pdr, avg_hops, int(sum(delivered))


def eval_random_on_graph(device, n_nodes, edge_prob, graph_seed, max_steps=30, n_eval=100):
    env = RoutingEnvV2(n_nodes=n_nodes, edge_prob=edge_prob, seed=graph_seed, max_steps=max_steps)
    rng = np.random.RandomState(graph_seed + 500000)
    delivered = []
    for _ in range(n_eval):
        env.reset()
        done, steps = False, 0
        while not done:
            mask = env.get_action_mask()
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps >= max_steps:
                break
        delivered.append(1 if env.current_node == env.destination else 0)
    return 100.0 * np.mean(delivered)


def append_rows(rows, path):
    """Append rows to CSV incrementally -- creates the file+header on first
    write, appends without header afterward. Never overwrites prior data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not file_exists, index=False)


def already_done_pairs(path):
    """Returns the set of (model, train_seed) pairs already fully present,
    so a rerun after a crash skips them instead of redoing the work."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    # a (model, seed) pair is "done" only if it has all 8 configs recorded
    counts = df.groupby(["model", "train_seed"]).size()
    n_configs = len(ALL_CONFIGS) + len(TIGHT_BUDGET_CONFIGS)
    done = counts[counts >= n_configs].index.tolist()
    return set(done)


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    done = already_done_pairs(CSV_PATH)
    if done:
        print(f"Resuming -- {len(done)} (model, seed) pairs already complete, skipping those.")

    for model_name in ["gnn_dqn_v2", "dqn_v2"]:
        for seed in ALL_SEEDS:
            if (model_name, seed) in done:
                print(f"--- {model_name} seed={seed}: already done, skipping ---")
                continue

            print(f"--- {model_name} seed={seed}: training (3000 episodes) ---")
            result, extras = train_one_seed(
                model_name, seed=seed, episodes=TRAIN_EPISODES,
                device=device, verbose=False, return_model=True,
            )
            policy = extras["model"]

            seed_rows = []
            for cfg in ALL_CONFIGS:
                pdr, avg_hops, n_delivered = eval_policy_on_graph(
                    policy, device, cfg["n_nodes"], cfg["edge_prob"], cfg["graph_seed"],
                )
                print(f"    [{cfg['label']}] PDR={pdr:.1f}%  avg_hops={avg_hops:.2f}")
                seed_rows.append({
                    "model": model_name, "train_seed": seed, "config": cfg["label"],
                    "n_nodes": cfg["n_nodes"], "edge_prob": cfg["edge_prob"],
                    "pdr": pdr, "avg_hops": avg_hops, "n_delivered": n_delivered,
                    "max_steps": 30,
                })

            for cfg in TIGHT_BUDGET_CONFIGS:
                pdr, avg_hops, n_delivered = eval_policy_on_graph(
                    policy, device, cfg["n_nodes"], cfg["edge_prob"], cfg["graph_seed"],
                    max_steps=HALVED_MAX_STEPS,
                )
                print(f"    [{cfg['label']}, max_steps={HALVED_MAX_STEPS}] PDR={pdr:.1f}%  avg_hops={avg_hops:.2f}")
                seed_rows.append({
                    "model": model_name, "train_seed": seed, "config": cfg["label"] + "_halved_budget",
                    "n_nodes": cfg["n_nodes"], "edge_prob": cfg["edge_prob"],
                    "pdr": pdr, "avg_hops": avg_hops, "n_delivered": n_delivered,
                    "max_steps": HALVED_MAX_STEPS,
                })

            # SAVE IMMEDIATELY after this seed, not at the end of the whole run
            append_rows(seed_rows, CSV_PATH)
            print(f"    -- saved seed {seed} to {CSV_PATH}")

    # --- random-policy baseline (only needs to run once, not per seed) ---
    if not os.path.exists(RANDOM_CSV_PATH):
        print("\n--- random-policy baselines ---")
        rand_rows = []
        for cfg in ALL_CONFIGS:
            pdr = eval_random_on_graph(device, cfg["n_nodes"], cfg["edge_prob"], cfg["graph_seed"])
            print(f"    [{cfg['label']}] random PDR={pdr:.1f}%")
            rand_rows.append({"config": cfg["label"], "max_steps": 30, "random_pdr": pdr})
        for cfg in TIGHT_BUDGET_CONFIGS:
            pdr = eval_random_on_graph(device, cfg["n_nodes"], cfg["edge_prob"], cfg["graph_seed"], max_steps=HALVED_MAX_STEPS)
            print(f"    [{cfg['label']}_halved_budget] random PDR={pdr:.1f}%")
            rand_rows.append({"config": cfg["label"] + "_halved_budget", "max_steps": HALVED_MAX_STEPS, "random_pdr": pdr})
        pd.DataFrame(rand_rows).to_csv(RANDOM_CSV_PATH, index=False)
        print(f"Saved {RANDOM_CSV_PATH}")
    else:
        print(f"\n{RANDOM_CSV_PATH} already exists, skipping random baseline.")

    # --- final summary ---
    df = pd.read_csv(CSV_PATH)
    print("\n=== Row count check ===")
    print(df.groupby("model")["train_seed"].nunique(), "seeds per model (should be 10 each)")
    print("\n=== Summary: mean PDR by config x model ===")
    summary = df.groupby(["config", "model"])["pdr"].agg(["mean", "std"]).unstack()
    print(summary)


if __name__ == "__main__":
    main()
