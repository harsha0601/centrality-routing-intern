# Paper 1 Summary — Centrality-Based Routing in Networks

**Paper:** Wan, Z., Mahajan, Y., Kang, B. W., Moore, T. J., & Cho, J. H. (2021).
"A Survey on Centrality Metrics and Their Network Resilience Analysis."
*IEEE Access*, Vol. 9, pp. 104773–104819.
DOI: 10.1109/ACCESS.2021.3094194
Free PDF: https://arxiv.org/pdf/2011.14575

---

## 1. Problem

Traditional network routing and resilience analysis relies on a small set of
well-known centrality metrics — primarily degree and betweenness — chosen by
convention rather than by evidence of their suitability. This creates two
related problems: (a) routing protocols that use a single centrality signal
may select sub-optimal relay nodes, and (b) network designers cannot
accurately assess how vulnerable a topology is to targeted attacks without
understanding which nodes are truly "central" under different definitions.

The paper asks: among the full landscape of centrality metrics developed
across decades of network science, which ones are most useful for
communication networks — specifically for routing decisions, relay selection,
and resilience evaluation?

---

## 2. Centrality Metrics Surveyed

The paper surveys and classifies **over 30 centrality metrics** across several
categories. The most relevant to routing in wireless/opportunistic networks are:

| Metric | Definition | Routing Relevance |
|---|---|---|
| **Degree Centrality** | Fraction of neighbours a node has | Identifies highly-connected nodes; fast to compute |
| **Betweenness Centrality** | Fraction of all-pairs shortest paths passing through a node | Primary routing signal — high-betweenness nodes are natural relays |
| **Closeness Centrality** | Inverse of average shortest path length to all other nodes | Measures how quickly a node can reach others — relevant for delay minimisation |
| **Eigenvector Centrality** | A node's importance weighted by its neighbours' importance (iterative) | Captures influence propagation; useful for epidemic/broadcast routing |
| **PageRank** | Variant of eigenvector with damping factor | Used in social opportunistic routing (e.g. PeopleRank) |
| **K-Shell / K-Core** | Coreness of a node within progressively pruned sub-graphs | Identifies stable backbone nodes for infrastructure-like relay sets |

The paper notes that betweenness and degree dominate current use "because they
are popular, not because they are always the most appropriate metric" — a key
gap the authors highlight.

---

## 3. Routing Logic

The survey maps centrality metrics to routing strategies across three network
types — communication, social, and contact networks — and synthesises the
following routing logic:

**Step 1 — Graph Construction:** Model the network as a graph G = (V, E) where
nodes are devices and edges represent contacts (or probabilities of contact in
opportunistic settings).

**Step 2 — Metric Computation:** Compute the chosen centrality metric(s) for
all nodes. For dynamic networks, this is done on a time-aggregated contact
graph or a sliding window.

**Step 3 — Relay Selection:** When a carrier node meets a candidate relay,
forward the message if the relay's centrality score exceeds the carrier's
score. This is the core forwarding rule used in protocols like SimBet
(betweenness + similarity) and BUBBLE Rap (local/global popularity).

**Step 4 — Threshold / Budget Control:** To limit overhead, a TTL (time-to-live)
or replication budget is applied. Nodes with higher centrality receive longer
TTL, allowing messages to persist longer in the network.

**Key insight from the survey:** Betweenness centrality is the strongest single
predictor of relay utility in sparse, intermittently connected networks because
it directly measures bridging potential. However, it is computationally
expensive (O(VE) for unweighted graphs), which motivates approximation methods
for real-time use.

---

## 4. Open Gaps Identified

The paper explicitly identifies the following research gaps relevant to our
project:

1. **Single-metric dominance:** Most routing protocols use only one centrality
   metric. Jointly optimising across multiple metrics (e.g. betweenness +
   closeness) remains largely unexplored for opportunistic networks.

2. **Static graph assumption:** Most centrality-based routing studies compute
   metrics on static graphs. In real opportunistic networks, contact topology
   changes continuously. Temporal centrality — computed on time-varying
   contact graphs — is under-studied.

3. **Scalability of betweenness:** Exact betweenness requires global graph
   knowledge, which is unavailable in distributed opportunistic settings.
   Distributed approximation algorithms are an open problem.

4. **Metric selection justification:** Papers rarely justify why they chose a
   specific centrality metric. A systematic, evidence-based comparison of
   metric choices under different mobility models and traffic patterns is
   missing.

5. **Disconnected graph handling:** Eigenvector centrality and some closeness
   variants are undefined or unstable on disconnected graphs — a common
   condition in opportunistic networks. Robust alternatives are needed.

---

## 5. Relevance to This Project

This survey directly motivates our research question: *can a multi-metric
centrality routing strategy outperform single-metric baselines (SimBet,
PROPHET) in opportunistic networks?*

Gap 1 is our primary research gap. Gaps 2 and 3 define the scope boundary
(static synthetic graphs this week; temporal real traces from Week 2).
Gap 5 was already encountered in our own notebook (eigenvector convergence
error on disconnected instances), validating the survey's findings.

The betweenness centrality identified in Section 2 is our primary routing
signal for Week 2's relay selection function in `src/routing.py`.

---
