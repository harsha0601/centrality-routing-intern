# Paper Outline — Centrality-Based Routing in Opportunistic Networks

## Target Venue
IEEE GLOBECOM 2025 / IEEE ICC 2026 (6-page conference paper)

## Title
GNN-DQN: A Graph Neural Network Deep Q-Network Agent for
Centrality-Guided Routing in Opportunistic Networks

## Authors
Gopu Devi Saranya, Aluru Ganesh Reddy, I.V.G. Harsha Vardhan
(Supervisor: Dr. Anil Carie, SRM University AP)

## Section Outline

### Abstract (150 words)
- Problem: opportunistic routing, no stable paths
- Gap: static heuristics cannot adapt
- Method: GNN-DQN combining structural + dynamic features
- Result: 81% PDR, 2.0 hops, +11.2% over best baseline
- Generalisation: 73% PDR on unseen 80-node graph

### 1. Introduction (400 words)
- Routing challenge in opportunistic networks
- Why centrality heuristics fail (topology-dependent, static)
- Why ML baseline insufficient (no online adaptation)
- Our approach: GNN-DQN
- 3 contributions (bullet points)

### 2. Related Work (400 words)
- Use paper/related_work.md directly

### 3. Methodology (800 words)
- Network model
- Centrality heuristics (BCR/DCR/CCR)
- Random Forest baseline
- GCN encoder design
- DQN agent + reward function
- Training procedure

### 4. Results (600 words)
- Table: 5-method KPI comparison
- Figure 1: PDR with error bars
- Figure 2: Hop count
- Figure 3: Generalisation
- Statistical analysis (p=0.065 trend)

### 5. Conclusion (200 words)
- Summary of findings
- Limitations (variance, generalisation gap)
- Future work (multi-topology training, GAT, TTL)