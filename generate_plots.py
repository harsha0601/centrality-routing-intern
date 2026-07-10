import matplotlib.pyplot as plt
import numpy as np

# Set style for academic plotting
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12})
colors = ['#7f7f7f', '#a6c8e0', '#c6dbef', '#9ecae1', '#1f77b4'] # Grey for baselines, Blue for Ours

# -----------------------------------------------------------------
# FIGURE 1: Packet Delivery Ratio (PDR) Comparison
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
methods = ['BCR', 'DCR', 'CCR', 'Random Forest', 'GNN-DQN (Ours)']
pdr_values = [66.70, 67.07, 62.43, 69.80, 81.00]
# Adding dummy error bars to represent the standard deviations mentioned in your text
pdr_errors = [0, 0, 0, 6.68, 11.33] 

bars = ax.bar(methods, pdr_values, yerr=pdr_errors, color=colors, edgecolor='black', capsize=5, width=0.6)
ax.set_ylabel('Packet Delivery Ratio (%)')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
# Improve x-axis label layout to avoid overlap
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('fig1_pdr_comparison.png', dpi=300)
plt.close()

# -----------------------------------------------------------------
# FIGURE 2: Average Hop Count Comparison
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
hop_values = [17.39, 17.46, 16.61, 18.09, 2.00]

bars = ax.bar(methods, hop_values, color=colors, edgecolor='black', width=0.6)
ax.set_ylabel('Average Hop Count')
ax.set_ylim(0, 22)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

# Improve x-axis label layout to avoid overlap
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('fig2_hop_comparison.png', dpi=300)
plt.close()

# -----------------------------------------------------------------
# FIGURE 3: Topology Generalisation
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4.5))
labels = ['Training Topology\n(50 Nodes)', 'Unseen Topology\n(80 Nodes)']
gen_pdr = [81.00, 73.00]

bars = ax.bar(labels, gen_pdr, color=['#1f77b4', '#aec7e8'], edgecolor='black', width=0.5)
ax.set_ylabel('Packet Delivery Ratio (%)')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_generalisation.png', dpi=300)
plt.close()

print("All 3 figures generated successfully!")