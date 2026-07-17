import matplotlib.pyplot as plt
import numpy as np

# --- Set Publication Style (IEEE standard) ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300
})

# ==========================================
# FIGURE 1: PDR Comparison (6 Methods)
# ==========================================
methods = ['BCR', 'DCR', 'CCR', 'Random\nForest', 'DQN\n(no GCN)', 'GNN-DQN\n(Ours)']

# Using your latest verified numbers from the Jupyter run
pdr_means = [66.70, 67.07, 62.43, 60.80, 88.70, 70.60]
pdr_stds = [0, 0, 0, 6.68, 6.53, 11.33] # Error bars for RF, DQN, GNN-DQN

fig, ax = plt.subplots(figsize=(8, 5))
# Colors: Blue for heuristics, Orange for ML, Green for Ablation, Red for Ours
colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax.bar(methods, pdr_means, yerr=pdr_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels on top of bars
for bar, mean in zip(bars, pdr_means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5, 
            f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Packet Delivery Ratio (%)')
ax.set_ylim(0, 110) # Increased ylim so the 88.7% error bar doesn't get cut off
ax.set_title('Figure 1: PDR Comparison Across Methods (with Ablation)')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig1_pdr_comparison.png', dpi=300)
print("Saved fig1_pdr_comparison.png")
plt.show()

# ==========================================
# FIGURE 2: Hop Count Comparison (6 Methods)
# ==========================================
hop_means = [17.39, 17.46, 16.61, 18.09, 2.38, 2.00]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax.bar(methods, hop_means, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels
for bar, mean in zip(bars, hop_means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
            f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Average Hop Count')
ax.set_ylim(0, 22)
ax.set_title('Figure 2: Average Hop Count Comparison (with Ablation)')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig2_hop_comparison.png', dpi=300)
print("Saved fig2_hop_comparison.png")
plt.show()
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