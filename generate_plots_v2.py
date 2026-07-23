"""
Regenerates ALL publication figures from the current, frozen CSVs.
The old figures (fig1/2/3) were from the pre-revision architecture
(81% PDR, single-graph 73% generalisation) and are now stale/wrong --
this script replaces them entirely with figures matching the actual
final result (flag-design fleet, 8 stress configs, generalisation).

Usage: python generate_plots_v2.py
Output: figures/fig1_pdr_comparison.png
        figures/fig2_hop_comparison.png
        figures/fig3_generalisation.png
        figures/fig4_stress_eval.png
        figures/fig5_dqn_vs_random.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def fig1_pdr_comparison():
    """All methods, 10-seed means, bar chart with error bars."""
    bcr_dcr_ccr = pd.read_csv("results/baseline_kpis_10seed.csv")
    rf = pd.read_csv("results/random_forest_10seed.csv")
    gnn_pooled = pd.read_csv("results/gnn_dqn_pooled_regenerated_10seed.csv")
    dqn_pooled = pd.read_csv("results/ablation_dqn_no_gcn.csv")
    gnn_v2 = pd.read_csv("results/gnn_dqn_v2_10seed.csv")
    dqn_v2 = pd.read_csv("results/dqn_v2_10seed.csv")

    methods, means, stds = [], [], []
    for name, sub in [("BCR", None), ("DCR", None), ("CCR", None)]:
        vals = bcr_dcr_ccr[bcr_dcr_ccr.method == name].groupby("seed")["delivered"].mean() * 100
        methods.append(name); means.append(vals.mean()); stds.append(vals.std())
    methods.append("Random\nForest"); means.append(rf["pdr"].mean()); stds.append(rf["pdr"].std())
    methods.append("GNN-DQN\n(pooled)"); means.append(gnn_pooled["pdr"].mean()); stds.append(gnn_pooled["pdr"].std())
    methods.append("Plain DQN\n(pooled)"); means.append(dqn_pooled["pdr"].mean()); stds.append(dqn_pooled["pdr"].std())
    methods.append("DQN-v2\n(per-node)"); means.append(dqn_v2["pdr"].mean()); stds.append(dqn_v2["pdr"].std())
    methods.append("GNN-DQN-v2\n(per-node,\nours)"); means.append(gnn_v2["pdr"].mean()); stds.append(gnn_v2["pdr"].std())

    colors = ["#8899aa"]*3 + ["#8899aa", "#d98c3f", "#d9736f", "#7f9fd9", "#3f8c5c"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{m:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_title("PDR Comparison Across All Methods (10-seed mean \u00B1 std)")
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig("figures/fig1_pdr_comparison.png")
    plt.close()
    print("Saved figures/fig1_pdr_comparison.png")


def fig2_hop_comparison():
    gnn_v2 = pd.read_csv("results/gnn_dqn_v2_10seed.csv")
    dqn_v2 = pd.read_csv("results/dqn_v2_10seed.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ["DQN-v2", "GNN-DQN-v2"]
    means = [dqn_v2["avg_hops"].mean(), gnn_v2["avg_hops"].mean()]
    stds = [dqn_v2["avg_hops"].std(), gnn_v2["avg_hops"].std()]
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=["#d9736f", "#3f8c5c"], edgecolor="black")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Hop Count")
    ax.set_title("Hop Count: Training Topology (10-seed mean \u00B1 std)\nNo significant difference, p = 0.92")
    plt.tight_layout()
    plt.savefig("figures/fig2_hop_comparison.png")
    plt.close()
    print("Saved figures/fig2_hop_comparison.png")


def fig3_generalisation():
    gen = pd.read_csv("results/generalisation_v2.csv")
    gen_rand = pd.read_csv("results/random_baseline_training_and_generalisation.csv")

    sizes = sorted(gen["unseen_n_nodes"].unique())
    gnn_vals = [gen[(gen.model == "gnn_dqn_v2") & (gen.unseen_n_nodes == n)]["unseen_pdr"].values[0] for n in sizes]
    dqn_vals = [gen[(gen.model == "dqn_v2") & (gen.unseen_n_nodes == n)]["unseen_pdr"].values[0] for n in sizes]
    rand_vals = [gen_rand[gen_rand.config == f"generalisation_{n}node"]["random_pdr"].values[0] for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(sizes))
    width = 0.25
    ax.bar(x - width, rand_vals, width, label="Random policy", color="#aaaaaa", edgecolor="black")
    ax.bar(x, dqn_vals, width, label="DQN-v2", color="#d9736f", edgecolor="black")
    ax.bar(x + width, gnn_vals, width, label="GNN-DQN-v2", color="#3f8c5c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} nodes" for n in sizes])
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_title("Generalisation to Unseen Graph Sizes (standard step budget, no fallback)")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig("figures/fig3_generalisation.png")
    plt.close()
    print("Saved figures/fig3_generalisation.png")


def fig4_stress_eval():
    stress = pd.read_csv("results/stress_eval.csv")
    rand = pd.read_csv("results/stress_eval_random_baseline.csv")

    configs = stress["config"].unique().tolist()
    gnn_means = [stress[(stress.config == c) & (stress.model == "gnn_dqn_v2")]["pdr"].mean() for c in configs]
    dqn_means = [stress[(stress.config == c) & (stress.model == "dqn_v2")]["pdr"].mean() for c in configs]
    rand_vals = [rand[rand.config == c]["random_pdr"].values[0] if (rand.config == c).any() else np.nan for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(configs))
    width = 0.25
    ax.bar(x - width, rand_vals, width, label="Random policy", color="#aaaaaa", edgecolor="black")
    ax.bar(x, dqn_means, width, label="DQN-v2", color="#d9736f", edgecolor="black")
    ax.bar(x + width, gnn_means, width, label="GNN-DQN-v2", color="#3f8c5c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_title("Section 5 Stress Evaluation: All 8 Pre-registered Configs (10-seed mean)")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig("figures/fig4_stress_eval.png")
    plt.close()
    print("Saved figures/fig4_stress_eval.png")


def fig5_dqn_vs_random():
    """Highlights the 'DQN-v2 at/below random' finding specifically."""
    stress = pd.read_csv("results/stress_eval.csv")
    rand = pd.read_csv("results/stress_eval_random_baseline.csv")

    configs = stress["config"].unique().tolist()
    dqn_means = [stress[(stress.config == c) & (stress.model == "dqn_v2")]["pdr"].mean() for c in configs]
    rand_vals = [rand[rand.config == c]["random_pdr"].values[0] if (rand.config == c).any() else np.nan for c in configs]
    diff = [d - r for d, r in zip(dqn_means, rand_vals)]
    colors = ["#c0392b" if d <= 0 else "#3f8c5c" for d in diff]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([c.replace("_", " ") for c in configs], diff, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("DQN-v2 PDR \u2212 Random-policy PDR (percentage points)")
    ax.set_title("DQN-v2 vs Random-Policy Baseline, Per Stress Config\n(red = DQN-v2 at or below random chance)")
    plt.tight_layout()
    plt.savefig("figures/fig5_dqn_vs_random.png")
    plt.close()
    print("Saved figures/fig5_dqn_vs_random.png")


if __name__ == "__main__":
    fig1_pdr_comparison()
    fig2_hop_comparison()
    fig3_generalisation()
    fig4_stress_eval()
    fig5_dqn_vs_random()
    print("\nAll figures regenerated in figures/. Old figures (fig1/2/3 showing the")
    print("pre-revision 81%/73% numbers) should be considered stale -- these overwrite them.")