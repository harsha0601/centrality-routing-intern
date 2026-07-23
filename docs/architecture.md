# GNN-DQN Architecture

## Overview
GCN Encoder → Global Mean Pooling → 2-layer MLP → Q-values

## Input
- Node features: [degree, betweenness, closeness, traffic_load] → shape [50, 4]
- Edge index: [2, 358]

## GCN Encoder
- GCNConv(4 → 32) → ReLU
- GCNConv(32 → 16) → ReLU
- Output: node embeddings [50, 16]

## Global Pooling
- global_mean_pool over all nodes → graph embedding [1, 16]

## DQN Head (MLP)
- Linear(16 → 64) → ReLU
- Linear(64 → 50) → Q-values for each next-hop node

## Action
- argmax Q-values → next-hop node index

## Reward
- +10 - 0.1*delay on delivery
- -5 on timeout
- -1 on revisiting node
- -0.1*delay per normal hop