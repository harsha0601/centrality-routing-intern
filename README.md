# Centrality Routing Intern Project

## Project Scope
This project focuses on applying graph centrality metrics to routing strategies in wireless ad-hoc opportunistic networks. The goal is to develop a GNN + Deep Reinforcement Learning agent that outperforms classical centrality routing heuristics.

## Tools Used
- Python 3.11, NetworkX, Matplotlib, Pandas
- Jupyter Notebook, Git & GitHub, VS Code

---

## Week 1 — Environment Setup & Graph Fundamentals ✅
- Installed Python, Jupyter, Git, VS Code. Set up repo structure
- Generated 20-node Erdős–Rényi graph, computed 4 centrality metrics
- Key finding: Nodes 2 & 11 ranked #1 across all metrics
- Paper read: *Routing Strategy based on Centrality in Opportunistic Networks*

## Week 2 — Classical Centrality Routing Baselines ✅
- Built 50-node weighted network simulator
- Implemented BCR, DCR, CCR — ran 3 seeds x 1000 trials

| Method | PDR | Avg Delay | Avg Hops |
|--------|-----|-----------|----------|
| BCR | 66.70% | 97.41 ms | 17.39 |
| DCR | 67.07% | 90.90 ms | 17.46 |
| CCR | 62.43% | 84.74 ms | 16.61 |

- GNN-DQN target (Week 5): PDR > 75%, Delay < 84 ms
- Paper read: *GNN-Based Routing Optimization in Large-Scale IoT Deployments* — Babaria et al., 2025

## Week 3 — ML Baseline + Routing Environment ✅
- Generated 5000-sample labelled routing dataset (3 seeds)
- Trained Random Forest classifier as ML baseline
- Built custom Gymnasium routing environment (`envs/routing_env.py`)
- Tested random agent policy in the environment

## Results (Week 3)

| Method | PDR | Avg Delay | Avg Hops |
|--------|-----|-----------|----------|
| RF Baseline | 69.80% | 71.11 ms | 18.09 |
| Random Agent (Gym env) | 16% | - | - |

- RF beats all Week 2 heuristics on PDR and delay
- RF feature importance: traffic_load (60%) > betweenness (21%) > closeness (13%) > degree (5%)
- Random agent confirms environment works — GNN-DQN must beat 16% PDR by a wide margin

## Paper Read
- Paper 3: *Relational Deep Reinforcement Learning for Routing in Wireless Networks* — Manfredi et al., IEEE INFOCOM 2021

## Week 4 —## Week 4 — GNN Foundations ✅
- Installed PyTorch + PyTorch Geometric, verified with the Cora GCN example (23,063 params)
- Converted the 50-node network simulator graph to PyG format (50 nodes, 358 directed edges, 4 node features)
- Trained a 2-layer GCN (GCNConv 3→32→16) to predict betweenness centrality from [degree, closeness, traffic_load]
- Test MSE: 0.000055 — the GCN learns structural importance almost perfectly, even on unseen nodes
- t-SNE visualisation confirms high-betweenness nodes (36, 28, 39, 45, 29) and low-betweenness nodes (16, 4, 24, 18, 7) form visibly separate clusters in the learned embedding space

## Paper / Contribution
- Finalised research question — see `paper/contribution.md`
- Established the evidence trail: heuristics (62-67% PDR) → RF (69.8% PDR) → GCN structural embeddings (test MSE 0.000055) → GNN-DQN target (Week 5)
## Week 5 —🔜 Coming Next
## Week 6 —
## Week 7 —
## Week 8 —