#!/usr/bin/env python3
"""
run_sweep.py — hyperparameter sweeps for q_agent (week 4).

Sweeps any combination of the env-tunable hyperparameters wired into
train.py / callbacks.py:
    Q_AGENT_ALPHA, Q_AGENT_GAMMA, Q_AGENT_EPS_DECAY, Q_AGENT_PLAY_EPS
Each configuration trains FRESH through a curriculum, then evaluates.
Results accumulate in results/sweep_summary.csv; per-round curves land in
agent_code/q_agent/training_history.csv keyed by run_tag.

Usage examples (repo root):
    python run_sweep.py --param Q_AGENT_ALPHA --values 0.05 0.1 0.2 --seeds 2
    python run_sweep.py --param Q_AGENT_GAMMA --values 0.85 0.9 0.95 --seeds 2

DQN equivalents (lr, buffer size, target-sync, net width) should follow the
same pattern: read from env vars in dqn_agent/train.py, reuse this driver
with --agent dqn_agent once wired.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CURRICULUM = [
    # (scenario, extra agents, rounds, eps_start or None for fresh 1.0)
    ("coin-heaven", [], 400, None),
    ("loot-crate", [], 1200, "0.3"),
    ("classic", ["peaceful_agent"] * 3, 800, "0.2"),
    ("classic", ["coin_collector_agent"] * 3, 1500, "0.1"),
]
EVAL = ("classic", ["coin_collector_agent"] * 3, 20)


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:], file=sys.stderr)
        raise RuntimeError("failed: " + " ".join(cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="q_agent")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    agent_dir = Path("agent_code") / args.agent
    Path("results").mkdir(exist_ok=True)
    out = Path("results/sweep_summary.csv")
    new = not out.exists()
    with open(out, "a") as fsum:
        if new:
            fsum.write("param,value,seed,score,suicides,coins,table_size\n")

        for value in args.values:
            for seed in range(args.seeds):
                tag = f"{args.param}={value}_s{seed}"
                print(f"=== {tag} ===", flush=True)
                base = dict(os.environ, Q_AGENT_SEED=str(seed),
                            Q_AGENT_RUN_TAG=tag, **{args.param: str(value)})

                for model in agent_dir.glob("q_table_v3_*.pkl"):
                    model.unlink()

                for scenario, opponents, rounds, eps in CURRICULUM:
                    env = dict(base)
                    if eps is not None:
                        env["Q_AGENT_EPS_START"] = eps
                    run([sys.executable, "main.py", "play", "--agents",
                         args.agent, *opponents, "--train", "1",
                         "--scenario", scenario, "--n-rounds", str(rounds),
                         "--no-gui"], env)

                scenario, opponents, n = EVAL
                stats = f"results/{tag}_eval.json"
                run([sys.executable, "main.py", "play", "--agents",
                     args.agent, *opponents, "--scenario", scenario,
                     "--n-rounds", str(n), "--no-gui",
                     "--save-stats", stats], base)
                s = json.load(open(stats))["by_agent"][args.agent]
                import pickle
                tables = list(agent_dir.glob("q_table_v3_*.pkl"))
                size = len(pickle.load(open(tables[0], "rb"))) if tables else 0
                if tables:   # archive the exact model behind this number
                    shutil.copy(tables[0], f"results/{tag}_model.pkl")
                fsum.write(f"{args.param},{value},{seed},{s.get('score',0)},"
                           f"{s.get('suicides',0)},{s.get('coins',0)},{size}\n")
                fsum.flush()
                print(f"  score {s.get('score',0)}, suicides {s.get('suicides',0)}")


if __name__ == "__main__":
    main()
