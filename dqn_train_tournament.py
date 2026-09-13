#!/usr/bin/env python3
"""Curriculum training pipeline for dqn_agent.

One neural-network model is trained in stages: each stage starts from the
weights produced by the previous stage, so the order matters. Solo boards
teach navigation and safe bombing first; opponents are introduced afterwards.

This mirrors the RL-Team train_tournament.py curriculum, but uses the DQN
agent and PyTorch checkpoints.

    python dqn_train_tournament.py                 # full run
    python dqn_train_tournament.py --from-stage 3  # resume at stage 3
    python dqn_train_tournament.py --fresh         # delete the model and start over
    python dqn_train_tournament.py --dry-run       # print the plan only
    python dqn_train_tournament.py --scale 0.02    # quick plumbing check

The DQN model file is dqn_model_v5.pt.

Note:
    Each stage transfers the learned neural-network weights from the previous
    stage. The current DQN training implementation initializes a fresh replay
    buffer, optimizer, target network, and epsilon at the beginning of each
    main.py process.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


AGENT = "dqn_agent"
AGENT_DIR = Path("agent_code") / AGENT
MODEL = AGENT_DIR / "dqn_model_v5.pt"
CHECKPOINTS = Path("results/checkpoints")


# Same curriculum as RL-Team.
#
# (name, scenario, opponents, rounds, why)
STAGES = [
    (
        "1-coins",
        "coin-heaven",
        [],
        600,
        "navigation: walk to coins",
    ),
    (
        "2-crates",
        "loot-crate",
        [],
        4000,
        "bombing and escaping, solo",
    ),
    (
        "3-peaceful",
        "classic",
        ["peaceful_agent"] * 3,
        2000,
        "opponents that never bomb",
    ),
    (
        "4-collector",
        "classic",
        ["coin_collector_agent"] * 3,
        5000,
        "opponents that bomb for coins",
    ),
    (
        "5-rulebased",
        "classic",
        ["rule_based_agent"] * 3,
        5000,
        "the benchmark opponent",
    ),
]


# Evaluation after every stage.
EVAL_OPPONENTS = ["rule_based_agent"] * 3
EVAL_ROUNDS = 30


def run(cmd, env, label):
    """Run a subprocess and stop immediately if it fails."""

    print(
        f"    $ {' '.join(cmd)}",
        flush=True,
    )

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(
            result.stdout[-1500:],
            result.stderr[-1500:],
            file=sys.stderr,
        )
        raise SystemExit(
            f"stage {label} failed"
        )


def evaluate(tag):
    """Evaluate the current DQN model against three rule-based agents."""

    CHECKPOINTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = (
        f"results/eval_{tag}.json"
    )

    env = dict(
        os.environ,
        DQN_MODEL_PATH=str(MODEL),
    )

    run(
        [
            sys.executable,
            "main.py",
            "play",
            "--agents",
            AGENT,
            *EVAL_OPPONENTS,
            "--scenario",
            "classic",
            "--n-rounds",
            str(EVAL_ROUNDS),
            "--no-gui",
            "--save-stats",
            stats,
        ],
        env,
        tag,
    )

    data = json.load(
        open(stats)
    )["by_agent"]

    mine = data.get(
        AGENT,
        {},
    )

    others = [
        value.get("score", 0)
        for key, value in data.items()
        if key != AGENT
    ]

    survival = (
        (
            EVAL_ROUNDS
            - mine.get("suicides", 0)
        )
        / EVAL_ROUNDS
        * 100
    )

    print(
        f"    -> score {mine.get('score', 0):4d} "
        f"vs best opponent "
        f"{max(others) if others else 0:4d} | "
        f"coins {mine.get('coins', 0)} | "
        f"crates {mine.get('crates', 0)} | "
        f"survival {survival:.0f}%",
        flush=True,
    )

    checkpoint = (
        CHECKPOINTS
        / f"dqn_model_after_{tag}.pt"
    )

    shutil.copy(
        MODEL,
        checkpoint,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--from-stage",
        type=int,
        default=1,
        help="resume at this stage number (1-based)",
    )

    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply all round counts (0.02 = quick smoke test)",
    )

    ap.add_argument(
        "--fresh",
        action="store_true",
        help="delete the existing DQN model and start over",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only",
    )

    args = ap.parse_args()

    if args.fresh and MODEL.exists():
        MODEL.unlink()

        print(
            f"deleted {MODEL} -- starting from scratch",
            flush=True,
        )

    for number, (
        name,
        scenario,
        opponents,
        rounds,
        why,
    ) in enumerate(STAGES, 1):

        if number < args.from_stage:
            continue

        n = max(
            1,
            int(rounds * args.scale),
        )

        print(
            f"\n=== stage {number}/{len(STAGES)}: "
            f"{name} ({n} rounds) ===\n"
            f"    {why}",
            flush=True,
        )

        if args.dry_run:
            continue

        # The previous stage's dqn_model_v5.pt is loaded by train.py.
        #
        # DQN_MODEL_PATH is supplied explicitly so the model path is not
        # dependent on the current working directory or hard-coded behavior.
        env = dict(
            os.environ,
            DQN_MODEL_PATH=str(MODEL),
            DQN_RUN_TAG=f"stage{number}-{name}",
        )

        run(
            [
                sys.executable,
                "main.py",
                "play",
                "--agents",
                AGENT,
                *opponents,
                "--train",
                "1",
                "--scenario",
                scenario,
                "--n-rounds",
                str(n),
                "--no-gui",
            ],
            env,
            name,
        )

        evaluate(
            f"{number}-{name}"
        )

    if not args.dry_run:
        print(
            f"\nDone. Model: {MODEL}; "
            f"per-stage checkpoints: {CHECKPOINTS}/",
            flush=True,
        )


if __name__ == "__main__":
    main()
    