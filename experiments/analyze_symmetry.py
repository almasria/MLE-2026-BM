"""
analyze_symmetry.py — turn the ablation data into the report experiment.

Reads:
  agent_code/q_agent/training_history.csv   (written every round by train.py)
  results/ablation_summary.csv              (written by run_symmetry_ablation.py)
  results/dqn_benchmark.json                (optional; --save-stats of a dqn eval)

Produces:
  results/symmetry_ablation.png             (learning curves, seed bands)
  printed: rounds-to-convergence per condition, greedy results, table sizes,
           and the W2 milestone verdict.

METRIC (decide before looking at data, report verbatim):
  "Rounds to convergence" = first round at which the rolling mean (window
  W=100) of coins collected per TRAINING round reaches T=45 of 50.
  Both conditions share the identical epsilon schedule, so this compares
  learning speed under equal exploration. Caveat for the report: training-
  time coins are suppressed by exploration; an epsilon-free alternative is
  periodic greedy checkpointing, which costs extra evaluation runs.

Usage (repo root):  python analyze_symmetry.py [--window 100] [--threshold 45]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HISTORY = Path("agent_code/q_agent/training_history.csv")
SUMMARY = Path("results/ablation_summary.csv")
DQN_BENCH = Path("results/dqn_benchmark.json")
OUT_PNG = Path("results/symmetry_ablation.png")
COLS = ["run_tag", "symmetry", "round", "reward", "coins", "epsilon", "table_size"]


def rolling(x, w):
    """Rolling mean with a growing window at the start (no NaN padding)."""
    out = np.empty(len(x))
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=45.0)
    args = ap.parse_args()

    # ---- load per-round history, grouped by run ---------------------------
    runs = defaultdict(list)          # run_tag -> list of (round, coins)
    sym_of = {}
    with open(HISTORY) as f:
        for row in csv.reader(f):
            rec = dict(zip(COLS, row))
            runs[rec["run_tag"]].append((int(rec["round"]), float(rec["coins"])))
            sym_of[rec["run_tag"]] = int(rec["symmetry"])
    for tag in runs:
        runs[tag].sort()

    # ---- rounds-to-convergence per run ------------------------------------
    conv = defaultdict(list)          # symmetry -> list of rounds (or None)
    curves = defaultdict(list)        # symmetry -> list of rolling curves
    for tag, series in runs.items():
        coins = [c for _, c in series]
        roll = rolling(coins, args.window)
        curves[sym_of[tag]].append(roll)
        hit = np.argmax(roll >= args.threshold) if (roll >= args.threshold).any() else None
        conv[sym_of[tag]].append(None if hit is None else int(series[hit][0]))
        shown = "never" if hit is None else str(series[hit][0])
        print(f"{tag}: rounds-to-{args.threshold:g} = {shown}")

    print()
    for sym in (1, 0):
        vals = conv.get(sym, [])
        reached = [v for v in vals if v is not None]
        label = "ON " if sym else "OFF"
        if reached:
            mid = np.mean(reached)
            spread = (max(reached) - min(reached)) / 2 if len(reached) > 1 else 0
            extra = f" ({len(vals)-len(reached)} run(s) never reached)" if len(reached) < len(vals) else ""
            print(f"symmetry {label}: rounds-to-convergence {mid:.0f} +/- {spread:.0f}{extra}")
        elif vals:
            print(f"symmetry {label}: threshold never reached in any run")

    # ---- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {1: ("D4 canonicalization ON", "tab:blue"),
              0: ("no canonicalization", "tab:orange")}
    for sym, (label, color) in styles.items():
        cs = curves.get(sym)
        if not cs:
            continue
        n = min(len(c) for c in cs)
        arr = np.stack([c[:n] for c in cs])
        xs = np.arange(1, n + 1)
        ax.plot(xs, arr.mean(0), color=color, label=f"{label} ({arr.shape[0]} seed(s))")
        if arr.shape[0] > 1:
            ax.fill_between(xs, arr.min(0), arr.max(0), color=color, alpha=0.2)
    ax.axhline(args.threshold, ls="--", lw=1, color="gray")
    ax.text(5, args.threshold + 0.7, f"convergence threshold T={args.threshold:g}", fontsize=8, color="gray")
    ax.set_xlabel("training round")
    ax.set_ylabel(f"coins per round (rolling mean, W={args.window})")
    ax.set_title("Effect of D4 canonicalization on tabular Q-learning (coin-heaven)")
    ax.set_ylim(0, 52)
    ax.legend(loc="lower right")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nplot -> {OUT_PNG}")

    # ---- milestone verdict ------------------------------------------------
    print("\n--- W2 milestone check (>= 45/50 greedy on coin-heaven) ---")
    if SUMMARY.exists():
        with open(SUMMARY) as f:
            rows = list(csv.DictReader(f))
        sym_rows = [r for r in rows if r["symmetry"] == "1"]
        if sym_rows:
            g = [float(r["greedy_coins_per_round"]) for r in sym_rows]
            sizes = sorted({r["table_size"] for r in sym_rows})
            ok = "PASS" if np.mean(g) >= 45 else "FAIL"
            print(f"q_agent (sym): {np.mean(g):.1f} coins/round over {len(g)} seed(s), "
                  f"tables {sizes} -> {ok}")
    if DQN_BENCH.exists():
        d = json.load(open(DQN_BENCH))["by_agent"]["dqn_agent"]
        n = d.get("rounds", None)
        # save-stats stores totals; infer rounds from the eval you ran
        per = d.get("coins", 0) / (n if n else 20)
        print(f"dqn_agent: {per:.1f} coins/round -> {'PASS' if per >= 45 else 'FAIL'} "
              f"(assumes 20 eval rounds; adjust if different)")
    else:
        print("dqn_agent: run the benchmark first, e.g.:\n"
              "  python main.py play --agents dqn_agent --scenario coin-heaven "
              "--n-rounds 20 --no-gui --save-stats results/dqn_benchmark.json")


if __name__ == "__main__":
    main()
