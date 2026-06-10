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

## Week 3 — 🔜 Coming Next

## Week 4 —
## Week 5 —
## Week 6 —
## Week 7 —
## Week 8 —