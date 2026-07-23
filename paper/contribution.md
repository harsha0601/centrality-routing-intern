# Our Contribution

Our routing agent differs from prior work because it uses learned GNN
embeddings instead of hand-crafted centrality scores. While classical
heuristics (BCR, DCR, CCR) and the centrality survey (Wan et al. 2021)
rely on pre-computed, static centrality metrics, our GCN learns a
continuous 16-dimensional representation of each node's structural role
directly from the graph topology — achieving a test MSE of just 0.000055
when predicting betweenness centrality purely from local neighbourhood
features (degree, closeness, traffic_load).

This is significant because it shows the network's GNN encoder did not
just memorise the training nodes — it generalised the structural pattern
of "what makes a node a bridge" well enough to predict betweenness
accurately on 10 nodes it never saw during training. That generalisation
ability is the entire reason we are using a GNN instead of simply hard-coding
centrality scores: a GNN can be retrained or fine-tuned as the network
topology changes, while a manually computed centrality score has to be
recalculated from scratch every time.

This learned representation, combined with dynamic traffic features
(Week 3 showed traffic_load is the single most predictive routing feature,
at 60.3% importance — three times more predictive than betweenness itself),
will feed into a DQN agent in Week 5. This allows routing decisions to
jointly consider structure (like centrality) and real-time network state
(like congestion) — addressing the core limitation identified in Manfredi
et al. (2021): existing routing strategies do not generalise across diverse
network conditions because they encode fixed assumptions about either
topology or traffic, but rarely both together.

## Research Question (finalised)
Can a GNN-DQN agent that jointly learns from structural (centrality-like)
and dynamic (traffic) graph features outperform both classical centrality
heuristics and a supervised ML baseline (Random Forest) on packet delivery
ratio, end-to-end delay, and hop count — and does this advantage persist
when tested on unseen network topologies?

## Evidence so far
- Week 1-2: Centrality-based heuristics (BCR, DCR, CCR) achieve 62.43%-67.07% PDR,
  using only static structural information
- Week 3: Random Forest, combining structure + dynamic traffic features,
  improves PDR to 69.80% and cuts average delay from ~85-97ms down to 71.11ms
- Week 4: A 2-layer GCN learns the same structural signal (betweenness centrality)
  that Weeks 1-2 computed by hand, but as a continuous, generalisable embedding
  rather than a fixed formula — test MSE of 0.000055 confirms this signal is
  reliably learnable from local graph structure alone
- Week 5 target: A GNN-DQN agent that combines the GCN's structural embeddings
  with the traffic-awareness shown valuable in Week 3 should exceed 75% PDR
  and beat 71ms average delay — outperforming every baseline tested so far

## Why this matters for the final paper
This progression — from static heuristics, to a supervised ML baseline, to a
learned GNN representation — gives our paper a clean narrative: each week's
result motivates the next week's design choice, and the final GNN-DQN
comparison table will show a measurable, explainable improvement at each
stage rather than a single isolated number.