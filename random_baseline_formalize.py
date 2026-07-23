"""
Formalizes rows 7 and 10 of the provenance table: random-policy baseline on
the 50-node TRAINING topology and the 3 ORIGINAL generalisation graphs
(standard step budget) -- distinct from stress_eval's random baselines,
which only cover the 8 new stress configs.

Usage: python random_baseline_formalize.py
Output: results/random_baseline_training_and_generalisation.csv
"""
import numpy as np
import pandas as pd

from env_v2 import RoutingEnvV2

N_EVAL = 100


def eval_random_on_graph(n_nodes, edge_prob, graph_seed, max_steps=30, n_eval=N_EVAL):
    env = RoutingEnvV2(n_nodes=n_nodes, edge_prob=edge_prob, seed=graph_seed, max_steps=max_steps)
    rng = np.random.RandomState(graph_seed + 999000)
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


def main():
    rows = []

    # Row 7: training topology, 50 nodes, seed 42, standard budget
    print("--- Training topology (50 nodes, seed 42) ---")
    pdr = eval_random_on_graph(n_nodes=50, edge_prob=0.15, graph_seed=42)
    print(f"    random PDR = {pdr:.1f}%")
    rows.append({"config": "training_topology_50node", "n_nodes": 50, "edge_prob": 0.15,
                 "graph_seed": 42, "max_steps": 30, "random_pdr": pdr})

    # Row 10: original 3 generalisation graphs, standard budget (matches generalisation_test.py's UNSEEN_SPECS)
    gen_specs = [
        {"n_nodes": 80, "edge_prob": 0.12, "graph_seed": 99},
        {"n_nodes": 100, "edge_prob": 0.10, "graph_seed": 7},
        {"n_nodes": 65, "edge_prob": 0.13, "graph_seed": 13},
    ]
    for spec in gen_specs:
        label = f"generalisation_{spec['n_nodes']}node"
        print(f"--- {label} ---")
        pdr = eval_random_on_graph(spec["n_nodes"], spec["edge_prob"], spec["graph_seed"])
        print(f"    random PDR = {pdr:.1f}%")
        rows.append({"config": label, "n_nodes": spec["n_nodes"], "edge_prob": spec["edge_prob"],
                     "graph_seed": spec["graph_seed"], "max_steps": 30, "random_pdr": pdr})

    df = pd.DataFrame(rows)
    df.to_csv("results/random_baseline_training_and_generalisation.csv", index=False)
    print("\nSaved results/random_baseline_training_and_generalisation.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()