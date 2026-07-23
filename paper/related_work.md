# Related Work

## Theme 1 — Classical Centrality-Based Routing

Daly and Haahr (2007) proposed SimBet, which uses betweenness centrality
combined with social similarity to select relay nodes in delay-tolerant
networks. While SimBet improved delivery over epidemic routing, it requires
global graph knowledge and cannot adapt to dynamic topology changes.

Wan et al. (2021) surveyed over 30 centrality metrics and identified
betweenness as the strongest single routing signal in sparse networks.
Our Week 2 results partially contradicted this — DCR (degree routing)
achieved 67.07% PDR vs BCR (betweenness) at 66.70%, suggesting metric
effectiveness is topology-dependent.

## Theme 2 — Machine Learning Based Routing

Babaria et al. (2025) proposed GNN-ROF, a supervised GNN trained on
Dijkstra-optimal paths in IoT networks, achieving 96.5% PDR. However,
GNN-ROF requires retraining when conditions change and does not use
centrality metrics as features. Our Week 3 Random Forest baseline
achieved 69.8% PDR, with feature importance showing traffic_load (60.3%)
as the dominant predictor — far exceeding any static centrality score.

## Theme 3 — GNN and Deep RL Based Routing

Manfredi et al. (2021) developed a distributed DRL routing strategy using
relational features, demonstrating generalisation across diverse network
conditions. However, their approach uses fixed hand-crafted feature
aggregation rather than a learned GNN encoder, and does not use centrality
metrics as inputs.

Almasan et al. (2020) combined DRL and GNNs for centralised SDN routing
but did not target distributed opportunistic network settings. Our GNN-DQN
addresses these gaps: learned GCN encoder instead of fixed aggregation,
all 4 centrality scores as node features, and distributed operation.