"""
Single committed stats script. Recomputes every comparison/p-value used in
the manuscript directly from the tagged CSVs -- no number should be typed
by hand anywhere else. Run this any time to reproduce every statistic.

Usage: python stats_final.py
"""
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


def compare(name, gnn_vals, dqn_vals):
    print(f"\n--- {name} ---")
    print(f"GNN-DQN-v2: mean={sum(gnn_vals)/len(gnn_vals):.2f}  n={len(gnn_vals)}")
    print(f"DQN-v2:     mean={sum(dqn_vals)/len(dqn_vals):.2f}  n={len(dqn_vals)}")
    try:
        w = wilcoxon(gnn_vals, dqn_vals)
        t = ttest_rel(gnn_vals, dqn_vals)
        print(f"Wilcoxon p = {w.pvalue:.6f}")
        print(f"paired t p = {t.pvalue:.6e}")
    except Exception as e:
        print("stats error:", e)


def main():
    # --- Fleet: training-topology PDR, 10 seeds ---
    gnn_fleet = pd.read_csv("results/gnn_dqn_v2_10seed.csv").sort_values("seed")
    dqn_fleet = pd.read_csv("results/dqn_v2_10seed.csv").sort_values("seed")
    compare("Fleet PDR (training topology, 10 seeds)",
            gnn_fleet["pdr"].tolist(), dqn_fleet["pdr"].tolist())
    compare("Fleet avg_hops (training topology, 10 seeds)",
            gnn_fleet["avg_hops"].tolist(), dqn_fleet["avg_hops"].tolist())

    # --- Stress eval: per-config PDR, 10 seeds each ---
    stress = pd.read_csv("results/stress_eval.csv")
    for cfg in stress["config"].unique():
        g = stress[(stress.config == cfg) & (stress.model == "gnn_dqn_v2")].sort_values("train_seed")["pdr"].tolist()
        d = stress[(stress.config == cfg) & (stress.model == "dqn_v2")].sort_values("train_seed")["pdr"].tolist()
        compare(f"Stress eval: {cfg}", g, d)

    # --- Stress eval vs random baseline (DQN-v2 only, mean PDR per config) ---
    rand = pd.read_csv("results/stress_eval_random_baseline.csv")
    print("\n--- DQN-v2 vs random-policy baseline, per stress config ---")
    for cfg in stress["config"].unique():
        d_mean = stress[(stress.config == cfg) & (stress.model == "dqn_v2")]["pdr"].mean()
        r = rand[rand.config == cfg]["random_pdr"].values
        r_val = r[0] if len(r) else float("nan")
        flag = "AT/BELOW RANDOM" if d_mean <= r_val else "above random"
        print(f"{cfg}: DQN-v2 mean={d_mean:.1f}%  random={r_val:.1f}%  [{flag}]")

    # --- Generalisation (standard budget) ---
    gen = pd.read_csv("results/generalisation_v2.csv")
    print("\n--- Generalisation (standard budget) ---")
    print(gen[["model", "train_seed", "unseen_n_nodes", "unseen_pdr"]].to_string(index=False))
    gen_rand = pd.read_csv("results/random_baseline_training_and_generalisation.csv")
    print("\nRandom baselines (training topology + generalisation graphs):")
    print(gen_rand.to_string(index=False))

    # --- Pooled ablation (original architecture, regenerated) ---
    pooled_gnn = pd.read_csv("results/gnn_dqn_pooled_regenerated_10seed.csv").sort_values("seed")
    pooled_dqn = pd.read_csv("results/ablation_dqn_no_gcn.csv").sort_values("seed")
    compare("Pooled GNN-DQN vs Plain DQN (original architecture, regenerated)",
            pooled_gnn["pdr"].tolist(), pooled_dqn["pdr"].tolist())


if __name__ == "__main__":
    main()