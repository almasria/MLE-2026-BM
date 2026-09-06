#!/usr/bin/env python3
"""
train_tournament.py — the complete q_agent training pipeline.

ONE Q-TABLE, GROWN IN STAGES.  Every stage loads the table left by the
previous stage and adds to it, so the ORDER matters and the stages are not
alternatives.  Cheap solo boards teach navigation and safe bombing first;
opponents are introduced only once that base exists.

Run the whole thing (hours -- start it before bed):

    python train_tournament.py

Resume after an interruption (nothing is lost; the table is saved every round):

    python train_tournament.py --from-stage 3

See the plan without running it:

    python train_tournament.py --dry-run

Quick end-to-end check that the pipeline works (a few minutes, weak agent):

    python train_tournament.py --scale 0.02

After each stage the script evaluates the agent, prints a one-line report and
copies the table to results/checkpoints/, so a stage that makes things WORSE
can be rolled back by copying its predecessor over the model file.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

AGENT = "q_agent"
AGENT_DIR = Path("agent_code") / AGENT
# the model file follows the active feature version (Q_AGENT_FEATURE_VERSION)
_VERSION = os.environ.get("Q_AGENT_FEATURE_VERSION", "v5")
MODEL = AGENT_DIR / f"q_table_{_VERSION}_sym.pkl"
CHECKPOINTS = Path("results/checkpoints")

# (name, scenario, opponents, rounds, epsilon_start, why)
STAGES = [
    ("1-coins", "coin-heaven", [], 600, "1.0",
     "navigation only: walk to coins, no bombs needed"),
    ("2-crates", "loot-crate", [], 4000, "0.3",
     "bombing and escaping: the survival skills, learned cheaply alone"),
    ("3-peaceful", "classic", ["peaceful_agent"] * 3, 2000, "0.2",
     "other bodies on the board, but they never bomb: safe introduction"),
    ("4-collector", "classic", ["coin_collector_agent"] * 3, 5000, "0.1",
     "opponents that bomb for coins: real competition for space"),
    ("5-rulebased", "classic", ["rule_based_agent"] * 3, 5000, "0.05",
     "the benchmark opponent; the sheet says beating it is the bar"),
]

# Self-play is OPT-IN (--selfplay). Measured: 3000 rounds vs identical copies
# after the rule_based stage dropped the rule_based evaluation from 116 to 77.
# Copies never bomb when adjacent, so the aggression tuned to rule_based stops
# paying and the table drifts. If used, run it BEFORE stage 5, not after.
SELFPLAY_STAGE = ("6-selfplay", "classic", [AGENT] * 3, 3000, "0.1",
                  "sparring at your own level; only the FIRST instance trains")

# evaluation after every stage: always measured against the tournament setting
EVAL_OPPONENTS = ["rule_based_agent"] * 3
EVAL_ROUNDS = 30


def run(cmd, env, label):
    print(f"    $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:], result.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"stage {label} failed")


def evaluate(env, tag):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    stats = f"results/eval_{tag}.json"
    run([sys.executable, "main.py", "play", "--agents", AGENT, *EVAL_OPPONENTS,
         "--scenario", "classic", "--n-rounds", str(EVAL_ROUNDS), "--no-gui",
         "--save-stats", stats], env, tag)

    data = json.load(open(stats))["by_agent"]
    mine = data.get(AGENT, {})
    others = [v.get("score", 0) for k, v in data.items() if k != AGENT]
    import pickle
    size = len(pickle.load(open(MODEL, "rb"))) if MODEL.exists() else 0
    survival = (EVAL_ROUNDS - mine.get("suicides", 0)) / EVAL_ROUNDS * 100
    print(f"    -> score {mine.get('score', 0):4d} vs best opponent "
          f"{max(others) if others else 0:4d} | coins {mine.get('coins', 0)} | "
          f"crates {mine.get('crates', 0)} | survival {survival:.0f}% | "
          f"{size} states", flush=True)
    shutil.copy(MODEL, CHECKPOINTS / f"q_table_after_{tag}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", type=int, default=1,
                    help="resume at this stage number (1-based)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all round counts (0.02 = quick smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the existing table and start over")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfplay", action="store_true",
                    help="append the self-play stage (off by default, see note)")
    args = ap.parse_args()
    stages = STAGES + ([SELFPLAY_STAGE] if args.selfplay else [])

    if args.fresh and MODEL.exists():
        MODEL.unlink()
        print(f"deleted {MODEL} -- starting from an empty table")

    for number, (name, scenario, opponents, rounds, eps, why) in enumerate(stages, 1):
        if number < args.from_stage:
            continue
        n = max(1, int(rounds * args.scale))
        print(f"\n=== stage {number}/{len(stages)}: {name} ({n} rounds, "
              f"eps start {eps}) ===\n    {why}", flush=True)
        if args.dry_run:
            continue

        env = dict(os.environ, Q_AGENT_EPS_START=eps,
                   Q_AGENT_RUN_TAG=f"stage{number}-{name}")
        run([sys.executable, "main.py", "play", "--agents", AGENT, *opponents,
             "--train", "1", "--scenario", scenario,
             "--n-rounds", str(n), "--no-gui"], env, name)
        evaluate(dict(os.environ), f"{number}-{name}")

    if not args.dry_run:
        print("\nPipeline finished. The tournament model is "
              f"{MODEL}; per-stage checkpoints are in {CHECKPOINTS}/.")
        print("If a late stage made things worse, copy the better checkpoint "
              "back over the model file and retrain from there.")


if __name__ == "__main__":
    main()
