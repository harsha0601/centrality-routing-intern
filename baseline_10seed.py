# baseline_10seed.py
"""
Day-2 item: rerun BCR/DCR/CCR on all 10 pre-specified seeds. Straight
extension of the real methodology already in baseline_comparison.ipynb --
that code is genuine, it was just only run on 3 seeds (42, 7, 13).
"""
import networkx as nx
import numpy as np
import pandas as pd

SEEDS_10 = [42, 7, 13, 21, 99, 3, 17, 55, 8, 34]
NUM_TRIALS = 1000

def centrality_route(G, src, dst, centrality):
    if src == dst:
        return [src], 0
    path = [src]
    current = src
    visited = set([src])
    for _ in range(50):
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            return None, float("inf")
        next_hop = max(neighbors, key=lambda n: centrality[n])
        path.append(next_hop)
        visited.add(next_hop)
        if next_hop == dst:
            return path, len(path) - 1
        current = next_hop
    return None, float("inf")

def main():
    all_results = []
    for seed in SEEDS_10:
        np.random.seed(seed)
        G = nx.erdos_renyi_graph(50, 0.15, seed=seed)
        for u, v in G.edges():
            G[u][v]["latency"] = round(np.random.uniform(1, 10), 2)

        betweenness = nx.betweenness_centrality(G)
        degree = nx.degree_centrality(G)
        closeness = nx.closeness_centrality(G)
        nodes = list(G.nodes())

        for method, centrality in [("BCR", betweenness), ("DCR", degree), ("CCR", closeness)]:
            for _ in range(NUM_TRIALS):
                src = np.random.choice(nodes)
                dst = np.random.choice([n for n in nodes if n != src])
                path, hops = centrality_route(G, src, dst, centrality)
                delivered = path is not None
                delay = sum(G[path[j]][path[j + 1]]["latency"] for j in range(len(path) - 1)) if delivered else 0
                all_results.append({"seed": seed, "method": method, "delivered": delivered, "hops": hops, "delay": delay})
        print(f"seed {seed} done")

    df = pd.DataFrame(all_results)
    df.to_csv("results/baseline_kpis_10seed.csv", index=False)
    print(f"\nSaved results/baseline_kpis_10seed.csv -- total rows: {len(df)}")

    print("\n=== 10-seed summary (compare to old 3-seed: BCR=66.70, DCR=67.07, CCR=62.43) ===")
    for method in ["BCR", "DCR", "CCR"]:
        sub = df[df["method"] == method]
        per_seed_pdr = sub.groupby("seed")["delivered"].mean() * 100
        pdr_mean, pdr_std = per_seed_pdr.mean(), per_seed_pdr.std()
        delivered = sub[sub["delivered"]]
        avg_hops = delivered["hops"].mean()
        print(f"{method}: PDR = {pdr_mean:.2f}% +/- {pdr_std:.2f}   avg_hops = {avg_hops:.2f}")

if __name__ == "__main__":
    main()