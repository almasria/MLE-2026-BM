
"""
run_symmetry_ablation.py - D4 canonicalization ablation for q_agent.

For each condition (symmetry ON / OFF) x seed: train a FRESH agent, then
evaluate it greedily. Learning curves accumulate in
agent_code/q_agent/training_history.csv (columns: run_tag, symmetry, round,
reward, coins, epsilon, table_size); greedy results are printed and saved
to results/ablation_summary.csv.

Usage (from the repo root):
    python run_symmetry_ablation.py --seeds 3 --rounds 1000 --eval-rounds 20

Report figures to make from the outputs:
  1. Learning curves: avg coins vs round (rolling mean over 100), one line
     per condition, shaded band = spread over seeds.
  2. Final greedy coins/round: mean +/- spread per condition.
  3. Q-table size per condition (the compression factor).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

AGENT = "q_agent"
AGENT_DIR = Path("agent_code") / AGENT


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--eval-rounds", type=int, default=20)
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    summary = []

    for symmetry in (1, 0):
        for seed in range(args.seeds):
            tag = f"sym{symmetry}_seed{seed}"
            print(f"=== {tag}: training {args.rounds} rounds ===", flush=True)

            env = dict(os.environ,
                       Q_AGENT_SYMMETRY=str(symmetry),
                       Q_AGENT_SEED=str(seed),
                       Q_AGENT_RUN_TAG=tag)

            # fresh model per run (per-condition filenames set in callbacks.py)
            model = AGENT_DIR / ("q_table_sym.pkl" if symmetry else "q_table_plain.pkl")
            model.unlink(missing_ok=True)

            run([sys.executable, "main.py", "play", "--agents", AGENT,
                 "--train", "1", "--scenario", "coin-heaven",
                 "--n-rounds", str(args.rounds), "--no-gui"], env)

            stats = f"results/{tag}_eval.json"
            run([sys.executable, "main.py", "play", "--agents", AGENT,
                 "--scenario", "coin-heaven", "--n-rounds", str(args.eval_rounds),
                 "--no-gui", "--save-stats", stats], env)

            coins = json.load(open(stats))["by_agent"][AGENT].get("coins", 0)
            import pickle
            table_size = len(pickle.load(open(model, "rb")))
            row = (tag, symmetry, seed, coins / args.eval_rounds, table_size)
            summary.append(row)
            print(f"    -> {row[3]:.1f} coins/round greedy, {table_size} states", flush=True)

    with open("results/ablation_summary.csv", "w") as f:
        f.write("run_tag,symmetry,seed,greedy_coins_per_round,table_size\n")
        for r in summary:
            f.write(",".join(map(str, r)) + "\n")

    for sym in (1, 0):
        vals = [r[3] for r in summary if r[1] == sym]
        sizes = [r[4] for r in summary if r[1] == sym]
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / 2
        print(f"symmetry={'ON ' if sym else 'OFF'}: greedy coins/round "
              f"{mean:.1f} +/- {spread:.1f}  (tables: {sizes})")


if __name__ == "__main__":
    main()
