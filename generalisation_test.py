"""
Day-2 generalisation test, per the revision plan Section 3.2 / Day 2:
evaluate the frozen v2 policies (no fine-tuning) on three pre-specified
unseen graphs. Because v2 uses per-node masking, there is NO environment-level
fallback here -- this is the first honest generalisation result either
architecture has had. Do not add or swap graphs after seeing results.

Pre-specified spec (from the plan -- do not change):
    80 nodes, seed 99, p=0.12
    100 nodes, seed 7,  p=0.10
    65 nodes,  seed 13, p=0.13
"""
import numpy as np
import pandas as pd
import torch

from env_v2 import RoutingEnvV2
from models_v2 import masked_argmax
from train_v2 import train_one_seed, build_edge_index

UNSEEN_SPECS = [
    {"train_seed": 99, "n_nodes": 80, "edge_prob": 0.12},
    {"train_seed": 7,  "n_nodes": 100, "edge_prob": 0.10},
    {"train_seed": 13, "n_nodes": 65, "edge_prob": 0.13},
]

EPISODES = 3000  # match the fleet's final, converged episode count


def eval_on_graph(policy, device, n_nodes, edge_prob, unseen_seed_offset=100000, n_eval=100):
    """Evaluate a frozen policy on a fresh graph of size n_nodes. No fallback:
    masked_argmax only ever returns a valid neighbour index."""
    env = RoutingEnvV2(n_nodes=n_nodes, edge_prob=edge_prob, seed=unseen_seed_offset)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    for model_name in ["gnn_dqn_v2", "dqn_v2"]:
        for spec in UNSEEN_SPECS:
            train_seed = spec["train_seed"]
            print(f"--- {model_name}: training on seed={train_seed} (50 nodes), "
                  f"then evaluating frozen policy on unseen {spec['n_nodes']}-node graph ---")

            # Retrain this one seed (matches the fleet's frozen-weights-at-seed policy;
            # we didn't checkpoint weights during the fleet run, so this reproduces them
            # deterministically -- verified reproducible earlier in this project).
            result, extras = train_one_seed(
                model_name, seed=train_seed, episodes=EPISODES,
                device=device, verbose=False, return_model=True,
            )
            policy = extras["model"]

            pdr, avg_hops, n_delivered = eval_on_graph(
                policy, device, n_nodes=spec["n_nodes"], edge_prob=spec["edge_prob"]
            )
            print(f"    train-topology PDR (50 nodes): {result['pdr']:.2f}%")
            print(f"    unseen {spec['n_nodes']}-node PDR: {pdr:.2f}%  avg_hops={avg_hops:.2f}  "
                  f"n_delivered={n_delivered}/100")

            rows.append({
                "model": model_name,
                "train_seed": train_seed,
                "train_pdr_50node": result["pdr"],
                "unseen_n_nodes": spec["n_nodes"],
                "unseen_edge_prob": spec["edge_prob"],
                "unseen_pdr": pdr,
                "unseen_avg_hops": avg_hops,
                "unseen_n_delivered": n_delivered,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/generalisation_v2.csv", index=False)
    print("\nSaved results/generalisation_v2.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
