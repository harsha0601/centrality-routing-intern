# Gap Analysis — Paper 2

## Paper Details
Title: Graph Neural Network-Based Routing Optimization in Large-Scale IoT Deployments
Authors: Jigar Babaria, Kunvar Chokshi, Venkata Sai Abhishek Anala, Raunaq Malik,
         Reshma Thakkallapelly, Vikas Chaudhary
Conference: 2025 3rd International Conference on Advances in Computation,
            Communication and Information Technology (ICAICCIT)
DOI: 10.1109/ICAICCIT68829.2025.11434239

## What features/metrics does it use?
Node features: residual energy, link quality, buffer occupancy, hop distance.
Does NOT use betweenness, closeness, or eigenvector centrality as features —
uses only raw network state attributes instead of graph-theoretic centrality scores.

## What is its routing logic?
Models IoT topology as a dynamic weighted graph.
Uses multi-layer GNN message passing + graph attention to learn node embeddings.
For each node, predicts next hop that minimizes a weighted cost function
combining delay and energy consumption.
Trained end-to-end using cross-entropy loss + energy penalty + latency penalty.
No reinforcement learning — purely supervised, trained on known optimal paths.

## Key results from the paper
- PDR: 96.5% (SC-IoT), 95.2% (FIT-IoT) vs RPL baseline of 82.4%
- Latency: 102 ms vs RPL 158 ms (-22.3%)
- Energy: 0.88 mJ/packet vs RPL 1.24 mJ/packet (-18.9%)
- Scales to 10,000+ nodes with 3.1 ms inference on Jetson Nano

## Open Gaps identified

### Gap 1 — No Reinforcement Learning
GNN-ROF uses supervised learning trained on Dijkstra ground truth labels.
It cannot adapt to unseen traffic patterns at runtime without full retraining.
A DQN agent learns from reward signals continuously — no retraining needed.

### Gap 2 — No Centrality Metrics as Features
The paper uses raw node attributes (energy, buffer, link quality) but ignores
graph-theoretic centrality scores. Our Week 1 results showed that betweenness,
closeness and eigenvector centrality are strong predictors of relay utility
(Wan et al. 2021). Combining both raw features AND centrality scores as GNN
input should produce richer node embeddings.

### Gap 3 — Compared only against traditional protocols
GNN-ROF is benchmarked against RPL, LEACH, AODV, and Q-learning —
NOT against centrality-based heuristics (BCR, DCR, CCR).
Our project fills this gap by directly comparing GNN-DQN against
all three centrality heuristics we implemented in Week 2.

### Gap 4 — No ablation on centrality features
The paper's ablation (Table 4) tests attention, edge features, and energy loss —
but never tests whether adding betweenness centrality as a node feature
would improve routing decisions. This is our novel contribution.

## How our project addresses these gaps
- Gap 1: We add a DQN agent on top of GNN — learns routing policy via rewards
- Gap 2: We include all 4 centrality scores as node features in the GNN encoder
- Gap 3: We compare GNN-DQN directly against BCR, DCR, CCR baselines
- Gap 4: We run an ablation study: GNN-DQN with vs without centrality features

## Connection to our Week 2 results
Our BCR baseline achieved 66.7% PDR, CCR 62.43%, DCR 67.07%.
GNN-ROF achieved 96.5% PDR on a much larger network (10,000 nodes).
This confirms that GNN-based routing is significantly superior to heuristics —
our GNN-DQN agent must close this gap on our 50-node simulator.
Target for Week 5: PDR > 80%, delay < 85 ms (beat CCR's best delay of 84.74 ms).