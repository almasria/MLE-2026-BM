#!/usr/bin/env python3
"""Evaluate a Q-table under a fixed protocol and record the result.

    python scripts/evaluate.py --model results/candidates/tournament.pkl
    python scripts/evaluate.py --model results/candidates/tournament.pkl --opponents mixed --rounds 50
    python scripts/evaluate.py --model results/candidates/tournament.pkl --seeds 3

Opponent sets:
    rulebased   three rule_based_agent            (default; the benchmark)
    collector   three coin_collector_agent
    peaceful    three peaceful_agent
    mixed       rule_based + coin_collector + peaceful
    solo-crates loot-crate scenario, no opponents
    solo-coins  coin-heaven scenario, no opponents

Each run appends one row to results/ledger.csv and saves the raw stats JSON
next to it, so every number in the report can be traced to a file. The
evaluated model is never modified.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AGENTS = {
    "RL-Team": "Q_AGENT_MODEL_PATH",
    "dqn_agent": "DQN_MODEL_PATH",
}

OPPONENTS = {
    "rulebased": ("classic", ["rule_based_agent"] * 3),
    "collector": ("classic", ["coin_collector_agent"] * 3),
    "peaceful": ("classic", ["peaceful_agent"] * 3),
    "mixed": ("classic", ["rule_based_agent", "coin_collector_agent", "peaceful_agent"]),
    "solo-crates": ("loot-crate", []),
    "solo-coins": ("coin-heaven", []),
}
LEDGER = Path("results/ledger.csv")
HEADER = ("timestamp,model,opponents,rounds,seed,score,score_per_round,best_opp,"
          "coins,kills,suicides,survival_pct,ms_per_step,stats_file\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent",
        default="RL-Team",
        choices=AGENTS,
        help="agent to evaluate (default: RL-Team)",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="path to the model/checkpoint to evaluate",
    )
    ap.add_argument("--opponents", default="rulebased", choices=OPPONENTS)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1, help="repeat with seeds 0..n-1")
    ap.add_argument("--tag", default="", help="label added to the ledger row")
    args = ap.parse_args()

    model = Path(args.model).resolve()
    if not model.exists():
        sys.exit(f"model not found: {model}")
    scenario, opponents = OPPONENTS[args.opponents]
    Path("results/evaluations").mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(HEADER)

    for seed in range(args.seeds):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = f"{model.stem}_{args.agent}_{args.opponents}_r{args.rounds}_s{seed}_{stamp}"
        stats = f"results/evaluations/{label}.json"
        env = dict(
            os.environ,
            **{
                AGENTS[args.agent]: str(model),
                "Q_AGENT_SEED": str(seed),
                "DQN_SEED": str(seed),
            },
        )
        cmd = [sys.executable, "main.py", "play", "--agents", args.agent, *opponents,               "--scenario", scenario, "--n-rounds", str(args.rounds), "--no-gui",
               "--save-stats", stats]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-1500:], file=sys.stderr)
            sys.exit("evaluation failed")

        data = json.load(open(stats))["by_agent"]
        me = data[args.agent]
        opp = [v.get("score", 0) for k, v in data.items() if k != args.agent]
        n = args.rounds
        row = {
            "timestamp": stamp, "model": model.name, "opponents": args.opponents,
            "rounds": n, "seed": seed, "score": me.get("score", 0),
            "score_per_round": round(me.get("score", 0) / n, 3),
            "best_opp": max(opp) if opp else "", "coins": me.get("coins", 0),
            "kills": me.get("kills", 0), "suicides": me.get("suicides", 0),
            "survival_pct": round((n - me.get("suicides", 0)) / n * 100, 1),
            "ms_per_step": round(me.get("time", 0) / max(1, me.get("steps", 1)) * 1000, 2),
            "stats_file": stats,
        }
        with open(LEDGER, "a", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in HEADER.strip().split(",")])
        lead = (f"  ({(row['score'] / row['best_opp'] - 1) * 100:+.0f}% vs best opponent)"
                if opp and row["best_opp"] else "")
        print(f"{model.name} vs {args.opponents}, {n} rounds, seed {seed}: "
              f"score {row['score']} ({row['score_per_round']}/rd){lead} | "
              f"coins {row['coins']} kills {row['kills']} suicides {row['suicides']} "
              f"({row['survival_pct']}% survival) | {row['ms_per_step']} ms/step"
              + (f" [{args.tag}]" if args.tag else ""))


if __name__ == "__main__":
    main()
