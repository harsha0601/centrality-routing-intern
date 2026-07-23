# Paper 3 Summary — Relational Deep Reinforcement Learning for Routing

## Paper Details
Title: Relational Deep Reinforcement Learning for Routing in Wireless Networks
Authors: Victoria Manfredi, Alicia Wolfe, Bing Wang, Xiaolan Zhang
Conference: IEEE INFOCOM 2021
DOI: 10.1109/INFOCOM47145.2021.9469475

## Problem
Existing routing protocols are designed for specific network conditions
and fail when conditions change — e.g., a protocol for connected networks
fails when the network becomes disconnected. The paper asks: can we design
one routing algorithm that works across diverse traffic patterns, congestion
levels, connectivity, and link dynamics?

## What method does it use?
Deep Reinforcement Learning (DRL) with 3 key innovations:
1. Relational features — topology-independent node features
   (destination distance, queue length, degree, neighbour aggregates)
   so the DNN generalises to unseen network sizes and topologies
2. Packet-centric decisions — each packet is the RL agent,
   not the device. More natural reward propagation
3. Extended-time actions (options) — models queue waiting time
   as a single action, reducing training data needed

## Routing Logic
- State = packet features + device features + aggregated neighbour features
- Action = which neighbour to forward the packet to next
- Reward = 0 (delivered), -1 per hop (transition), large negative (dropped)
- DNN architecture: input → expansion (10F neurons) → compression (F/2) → Q-value
- Trained offline, then frozen and copied to all devices

## Key Results
- Outperforms Shortest Path (SP) and Backpressure (BP) on PDR and delay
- Generalises to networks up to 100 nodes despite training on 64 nodes
- Works on both connected and disconnected (delay-tolerant) topologies
- Delivers ~80% packets on disconnected networks where SP fails completely

## Open Gaps

### Gap 1 — No centrality features
Uses raw features (queue length, hop distance, degree) but completely
ignores betweenness, closeness, and eigenvector centrality.
Our Week 1 results and Wan et al. (2021) showed centrality scores are
strong predictors of relay utility — adding them should improve decisions.

### Gap 2 — No GNN encoder
Uses manual feature aggregation (min/mean/max of neighbours) instead of
a learned GNN message-passing encoder. A GNN would learn richer
structural embeddings automatically rather than relying on hand-crafted
aggregation rules.

### Gap 3 — Stationary devices only
Paper explicitly states "we assume devices are stationary — device
mobility leads to other interesting challenges, which we leave to
future work." Opportunistic networks have mobile nodes — a key gap.

### Gap 4 — No centrality-based relay pre-selection
The paper considers ALL neighbours as possible actions at every step.
Pre-filtering to high-centrality neighbours (as BCR/DCR/CCR do) could
reduce the action space and speed up convergence.

## How our project addresses these gaps
- Gap 1: We add all 4 centrality scores as node features in the GNN encoder
- Gap 2: We replace manual aggregation with a proper GCN/GAT encoder
- Gap 3: We model opportunistic (intermittently connected) networks
- Gap 4: We use centrality-guided action masking in the DQN policy

## Connection to our results
Our random agent got 16% PDR, heuristics got 62-67%, RF got 69.8%.
This paper's DRL gets ~80-100% on connected networks.
Our GNN-DQN target (Week 5): PDR > 75%, Delay < 84ms — combining
this paper's DRL approach with centrality features from Wan et al.