# Figure Captions

## Figure 1 — PDR Comparison
All five routing methods are compared on Packet Delivery Ratio (PDR)
across 5 random seeds × 200 trials, with error bars showing ± 1 std.
GNN-DQN achieves 81.0% PDR, outperforming the best baseline
(RF: 69.8%) by 11.2 percentage points, confirming the advantage of
online reinforcement learning over static and supervised routing strategies.

## Figure 2 — Hop Count Comparison
Average hop count across all five methods. GNN-DQN achieves 2.0 average
hops compared to 16–18 hops for all other methods — an 88.9% reduction —
demonstrating that the learned policy discovers dramatically shorter
delivery paths than any hand-crafted or supervised baseline.

## Figure 3 — Topology Generalisation
GNN-DQN PDR on the 50-node training graph vs an unseen 80-node graph
(different topology, never seen during training). A small drop confirms
partial generalisation; the gap, if any, motivates future work on
multi-topology training.