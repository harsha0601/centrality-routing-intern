"""
Day-1 GATE, per the mentor's plan: "if the smoke test fails by evening, STOP and
debug — do not launch the fleet on broken code."

=== INSTRUMENT CORRECTION #2, dated 2026-07-20 ===
DQN-v2 plateauing well below 100% at full convergence is NOT a gate failure --
it may be the experiment's answer (with only a binary destination flag and no
message passing, the MLP can only recognise the destination as a direct
neighbour; a persistent GNN-vs-MLP gap is what "message passing earns its
keep" looks like). The gate now distinguishes three situations instead of a
single pass/fail:

  1. CONVERGED (possibly to a lower asymptote): reward-curve flat over the
     final ~500 episodes, max-Q bounded and stable. --> PASS, this is a
     result, proceed to fleet, regardless of the exact PDR value or how it
     compares to the other model.
  2. NOT YET CONVERGED: reward still climbing over the final ~500 episodes.
     --> extend ONCE to 4000 episodes, both models (pre-committed, final
     extension -- if still climbing at 4000, escalate to the mentor rather
     than extending again).
  3. PATHOLOGY: max-Q diverging, OR reward collapses after having risen.
     --> HARD FAIL. Halt everything. Report before any further change.

Usage: python smoke_test.py [--episodes 150]
"""
import argparse
import time
import numpy as np
import torch

from train_v2 import train_one_seed

FINAL_EXTENSION_EPISODES = 4000  # pre-committed, one-time-only extension count


def _smooth(x, k):
    k = max(1, min(k, len(x)))
    if k <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def _analyze_reward_trend(rewards, window=500):
    window = min(window, len(rewards))
    tail = rewards[-window:]
    half = max(1, window // 2)
    first_half_mean = tail[:half].mean()
    second_half_mean = tail[-half:].mean()
    tail_std = tail.std() if tail.std() > 1e-6 else 1.0
    slope_ratio = (second_half_mean - first_half_mean) / tail_std  # rise in std units
    still_climbing = slope_ratio > 0.5

    smoothed = _smooth(rewards, k=max(1, len(rewards) // 20))
    peak = smoothed.max()
    end_val = smoothed[-1]
    # collapse: peak was meaningfully positive/high, and the run has since
    # given back a large fraction of that gain
    collapse = (peak > 0) and ((peak - end_val) > 0.5 * abs(peak))

    return {
        "slope_ratio": slope_ratio, "still_climbing": bool(still_climbing),
        "peak": float(peak), "end_val": float(end_val), "collapse": bool(collapse),
    }


def _analyze_maxq(maxqs):
    if len(maxqs) < 8:
        return {"maxq_ok": None, "trend": "insufficient data", "diverging": False}
    qn = len(maxqs) // 4
    mq1, mq2, mq3, mq4 = (maxqs[i*qn:(i+1)*qn].mean() for i in range(4))
    finite_ok = np.all(np.isfinite(maxqs))
    diverging = finite_ok and (mq4 > 1.5 * mq3) and (mq4 > 1.5 * mq2)
    trend = f"Q1={mq1:.2f} Q2={mq2:.2f} Q3={mq3:.2f} Q4={mq4:.2f}"
    return {"maxq_ok": bool(finite_ok and not diverging), "trend": trend, "diverging": bool(not finite_ok or diverging)}


def check(model_name, episodes, device):
    print(f"\n=== Smoke test: {model_name} ({episodes} episodes, seed 42) ===")
    t0 = time.time()
    result, extras = train_one_seed(
        model_name, seed=42, episodes=episodes, device=device,
        verbose=False, collect_diagnostics=True,
    )
    diag = extras["diagnostics"]
    elapsed = time.time() - t0

    rewards = np.array(diag["episode_rewards"])
    maxqs = np.array([q for q in diag["episode_maxq"] if not np.isnan(q)])

    trend = _analyze_reward_trend(rewards)
    maxq = _analyze_maxq(maxqs)
    pdr_ok = result["pdr"] > 60.0

    print(f"  wall time:        {elapsed:.1f}s for {episodes} episodes "
          f"(~{elapsed/episodes*1000:.0f} ms/episode)")
    print(f"  eval PDR:         {result['pdr']:.1f}%  avg_hops={result['avg_hops']:.2f}  "
          f"{'OK (>60%)' if pdr_ok else 'WATCH (<=60%)'}")
    print(f"  reward trend:     final-500 slope={trend['slope_ratio']:.2f} std-units  "
          f"{'STILL CLIMBING' if trend['still_climbing'] else 'FLAT (converged)'}"
          + (f"  | COLLAPSE DETECTED (peak={trend['peak']:.2f} -> end={trend['end_val']:.2f})" if trend["collapse"] else ""))
    print(f"  max-Q trend:      {maxq['trend']}  "
          f"{'OK (bounded/stabilising)' if maxq['maxq_ok'] else 'WATCH' if maxq['maxq_ok'] is None else 'DIVERGING'}")

    # --- three-way diagnosis, per the mentor's decision table ---
    if trend["collapse"] or maxq["diverging"]:
        diagnosis = "pathology"
    elif trend["still_climbing"]:
        diagnosis = "not_converged"
    else:
        diagnosis = "converged"

    label = {
        "pathology": "PATHOLOGY -- HARD FAIL, halt and report before any change",
        "not_converged": "NOT YET CONVERGED -- needs more episodes",
        "converged": "CONVERGED -- PASS (this is a result, regardless of the PDR value)",
    }[diagnosis]
    print(f"  --> {label}")

    return diagnosis, result["pdr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150,
                     help="Short run for a quick gate check, not the real 2000-episode result.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if str(device) == "cpu":
        print("WARNING: running on CPU. This smoke test will still work, but the full "
              "fleet run will be much faster on a GPU runtime "
              "(e.g. Colab: Runtime > Change runtime type > GPU).")

    diag_gnn, pdr_gnn = check("gnn_dqn_v2", args.episodes, device)
    diag_dqn, pdr_dqn = check("dqn_v2", args.episodes, device)

    print("\n" + "=" * 50)

    if diag_gnn == "pathology" or diag_dqn == "pathology":
        print("HARD FAIL -- pathology detected (max-Q diverging or reward collapsed).")
        print("Halt everything. Report to the mentor before any further change. Do not proceed.")

    elif diag_gnn == "not_converged" or diag_dqn == "not_converged":
        if args.episodes >= FINAL_EXTENSION_EPISODES:
            print(f"STILL CLIMBING at {args.episodes} episodes, which was the pre-committed "
                  f"final extension ({FINAL_EXTENSION_EPISODES}). Do NOT extend further on your "
                  f"own -- escalate this to the mentor for a decision.")
        else:
            print(f"NOT YET CONVERGED. Per the pre-committed rule, extend ONCE to "
                  f"{FINAL_EXTENSION_EPISODES} episodes for BOTH models:")
            print(f"  python smoke_test.py --episodes {FINAL_EXTENSION_EPISODES}")

    else:
        print("GATE PASSED -- both models converged (flat reward, bounded max-Q).")
        print(f"  gnn_dqn_v2 PDR: {pdr_gnn:.1f}%   dqn_v2 PDR: {pdr_dqn:.1f}%")
        if pdr_dqn < pdr_gnn - 5:
            print("  DQN-v2 converged to a meaningfully lower asymptote than GNN-DQN-v2 -- "
                  "per the mentor's framing, this IS a result, not a failure.")
        print(f"\nSafe to launch the full fleet at this episode count:")
        print(f"  python train_v2.py --episodes {args.episodes}")


if __name__ == "__main__":
    main()