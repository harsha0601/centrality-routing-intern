# random_forest_10seed.py
"""
Day-2 item: re-evaluate Random Forest under the same 100-episode protocol as
v2, across all 10 pre-specified seeds.

IMPORTANT: uses the REAL trained RandomForestClassifier from
rf_baseline.ipynb (trained once on results/routing_dataset.csv). This is
deliberately NOT the version in statistical_tests.ipynb, where code labelled
"RF PDR per seed" actually calls centrality_route(G, src, dst, bet) -- i.e.
computes the BETWEENNESS heuristic's result and reports it as RF.
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEEDS_10 = [42, 7, 13, 21, 99, 3, 17, 55, 8, 34]
N_EVAL_EPISODES = 100

def main():
    df = pd.read_csv("results/routing_dataset.csv")
    features = ["degree", "betweenness", "closeness", "traffic_load"]
    X = df[features]
    y = df["optimal_next_hop"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    print("RF trained on", len(df), "samples (seed-42 Dijkstra-labelled data)")

    rows = []
    for seed in SEEDS_10:
        np.random.seed(seed)
        G = nx.erdos_renyi_graph(50, 0.15, seed=seed)
        for u, v in G.edges():
            G[u][v]["latency"] = round(np.random.uniform(1, 10), 2)

        deg = nx.degree_centrality(G)
        bet = nx.betweenness_centrality(G)
        clo = nx.closeness_centrality(G)
        nodes = list(G.nodes())

        delivered_list, hop_list = [], []
        for _ in range(N_EVAL_EPISODES):
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            path = [src]
            current = src
            visited = {src}
            delivered = False
            for _ in range(50):
                neighbors = [n for n in G.neighbors(current) if n not in visited]
                if not neighbors:
                    break
                feat = pd.DataFrame([{
                    "degree": deg[current], "betweenness": bet[current],
                    "closeness": clo[current], "traffic_load": np.random.uniform(0, 1),
                }])
                next_hop = rf.predict(feat)[0]
                if next_hop not in neighbors:
                    next_hop = neighbors[0]
                path.append(next_hop)
                visited.add(next_hop)
                if next_hop == dst:
                    delivered = True
                    break
                current = next_hop
            delivered_list.append(1 if delivered else 0)
            hop_list.append(len(path) - 1)

        pdr = 100.0 * np.mean(delivered_list)
        avg_hops = float(np.mean([h for h, d in zip(hop_list, delivered_list) if d == 1])) if any(delivered_list) else float("nan")
        print(f"seed {seed}: PDR={pdr:.2f}%  avg_hops={avg_hops:.2f}")
        rows.append({"seed": seed, "pdr": pdr, "avg_hops": avg_hops})

    out = pd.DataFrame(rows)
    out.to_csv("results/random_forest_10seed.csv", index=False)
    print(f"\nSaved results/random_forest_10seed.csv")
    print(f"mean PDR = {out['pdr'].mean():.2f}% +/- {out['pdr'].std():.2f}")

if __name__ == "__main__":
    main()