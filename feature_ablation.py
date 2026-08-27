"""
feature_ablation.py

A lightweight experimental runner for feature comparisons.

This script does NOT change the feature code automatically. Instead, it forces
an isolated training/evaluation run for the CURRENT version of the feature
implementation in the repo.

Use it like this:

    # baseline feature version in the current code
    python3 feature_ablation.py --label baseline --rounds 1000 --eval-rounds 20

    # then edit features.py to the new feature version and run again
    python3 feature_ablation.py --label feature_v2 --rounds 1000 --eval-rounds 20

Each run resets the Q-model and training log before training, so it does NOT
continue from a previous trained checkpoint.

Outputs are saved under:

    evaluation_results/<timestamp>/<label>/

with:
- training_history.csv
- q_table_* .pkl (saved by q_agent)
- eval_<label>.json
- summary.csv
- metrics.png
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "agent_code" / "q_agent"
ACTIVE_FEATURES = AGENT_DIR / "features.py"
MODEL_FILES = [
    AGENT_DIR / "q_table_plain.pkl",
    AGENT_DIR / "q_table_sym.pkl",
    AGENT_DIR / "q_table_v3_plain.pkl",
    AGENT_DIR / "q_table_v3_sym.pkl",
]
TRAIN_HISTORY = AGENT_DIR / "training_history.csv"
OUT_ROOT = ROOT / "evaluation_results"


def reset_run_artifacts():
    for model_file in MODEL_FILES:
        if model_file.exists():
            model_file.unlink()
    if TRAIN_HISTORY.exists():
        TRAIN_HISTORY.unlink()


def activate_feature_file(feature_name: str):
    """Swap the active q_agent/features.py file to the selected source without
    changing the selected source itself.

    For legacy feature files such as featuresv2.py, we generate a small
    compatibility wrapper that preserves the old logic while expanding it to the
    v2/v3 tuple layout expected by q_agent.
    """
    if feature_name == "features.py":
        return

    source = Path(feature_name)
    if not source.is_absolute():
        source = AGENT_DIR / feature_name
    if not source.exists():
        raise FileNotFoundError(f"Feature file not found: {source}")

    if source.name == "featuresv2.py":
        wrapper = '''"""Auto-generated compatibility wrapper for the legacy baseline.

This file is generated at runtime by feature_ablation.py and is never written
back to the source baseline file. It preserves the legacy feature logic while
filling the v2/v3-compatible tuple shape expected by q_agent callbacks.
"""

from agent_code.q_agent.featuresv2 import state_to_features as legacy_state_to_features


def state_to_features(game_state):
    base = legacy_state_to_features(game_state)
    if base is None:
        return None

    # legacy baseline returns (obj, up, right, down, left)
    obj = base[0]
    nb = list(base[1:5])
    while len(nb) < 4:
        nb.append(0)

    # q_agent expects this v2+ prefix:
    # (objective_dir, up, right, down, left, urgency, safe_dir, bomb_safe,
    #  crates, opp_dir, opp_in_blast)
    urgency = 0
    safe_dir = 4
    bomb_safe = 1 if sum(nb) > 0 else 0
    crates = 0
    opp_dir = 4
    opp_in_blast = 0
    return (obj, *nb, urgency, safe_dir, bomb_safe, crates, opp_dir, opp_in_blast)
'''
        ACTIVE_FEATURES.write_text(wrapper)
        print(f"Activated compatibility wrapper for: {source.name}")
        return

    shutil.copy2(source, ACTIVE_FEATURES)
    print(f"Activated feature file: {source.name}")


def run_cmd(cmd, env=None):
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-2000:], file=sys.stderr)
        print(completed.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return completed


def summarize_eval_json(json_path: Path, agent_name: str | None = None):
    """Read a single evaluation JSON and return a compact row for the chosen agent."""
    with open(json_path) as f:
        data = json.load(f)

    by_agent = data.get("by_agent", {})
    if not by_agent:
        raise ValueError(f"No agent data in {json_path}")

    if agent_name is None:
        agent_name = next(iter(by_agent.keys()), None)
    if agent_name is None or agent_name not in by_agent:
        raise ValueError(f"Requested agent {agent_name!r} not found in {json_path}")

    a = by_agent[agent_name]
    rounds = float(a.get("rounds", 0) or 0)
    score = float(a.get("score", 0) or 0)
    coins = float(a.get("coins", 0) or 0)
    steps = float(a.get("steps", 0) or 0)
    kills = float(a.get("kills", 0) or 0)
    suicides = float(a.get("suicides", 0) or 0)

    return {
        "agent": agent_name,
        "rounds": rounds,
        "avg_reward": score / rounds if rounds else 0.0,
        "avg_coins": coins / rounds if rounds else 0.0,
        "avg_steps": steps / rounds if rounds else 0.0,
        "total_score": score,
        "kills": kills,
        "suicides": suicides,
        "coins_total": coins,
    }


def save_summary_csv(rows, path: Path):
    fieldnames = [
        "label",
        "agent",
        "rounds",
        "avg_reward",
        "avg_coins",
        "avg_steps",
        "total_score",
        "kills",
        "suicides",
        "coins_total",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="feature_run", help="experiment label, e.g. baseline or feature_v2")
    ap.add_argument("--feature-file", default="features.py", help="File under agent_code/q_agent to activate for this run, e.g. features.py or featuresv2.py")
    ap.add_argument("--agents", nargs="+", default=["q_agent"], help="Agents to deploy, e.g. --agents q_agent or --agents q_agent rule_based_agent")
    ap.add_argument("--train", type=int, default=1, help="How many of the first agents are in training mode")
    ap.add_argument("--rounds", type=int, default=1000, help="Training rounds")
    ap.add_argument("--eval-rounds", type=int, default=20, help="Evaluation rounds")
    ap.add_argument("--scenario", default="coin-heaven", choices=["empty", "coin-heaven", "loot-crate", "classic"])
    ap.add_argument("--seed", type=int, default=0, help="Seed for this run")
    ap.add_argument("--symmetry", type=int, default=1, choices=[0, 1], help="1 for D4 symmetry on, 0 off")
    ap.add_argument("--no-gui", action="store_true", help="run without GUI")
    args = ap.parse_args()

    activate_feature_file(args.feature_file)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUT_ROOT / stamp / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    reset_run_artifacts()

    env = os.environ.copy()
    env["Q_AGENT_SYMMETRY"] = str(args.symmetry)
    env["Q_AGENT_SEED"] = str(args.seed)
    env["Q_AGENT_RUN_TAG"] = f"{args.label}_seed{args.seed}"

    print(f"[1/2] Training {args.label} for {args.rounds} rounds...")
    train_cmd = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        *args.agents,
        "--train",
        str(args.train),
        "--scenario",
        args.scenario,
        "--n-rounds",
        str(args.rounds),
        "--no-gui",
    ]
    run_cmd(train_cmd, env=env)

    training_copy = run_dir / "training_history.csv"
    if TRAIN_HISTORY.exists():
        training_copy.write_text(TRAIN_HISTORY.read_text())

    eval_json = run_dir / f"eval_{args.label}.json"
    print(f"[2/2] Evaluating {args.label} for {args.eval_rounds} rounds...")
    eval_cmd = [
        sys.executable,
        "main.py",
        "play",
        "--agents",
        *args.agents,
        "--scenario",
        args.scenario,
        "--n-rounds",
        str(args.eval_rounds),
        "--no-gui",
        "--save-stats",
        str(eval_json),
    ]
    run_cmd(eval_cmd, env=env)

    selected_agent = args.agents[0] if args.agents else None
    if args.train > 0 and args.agents:
        selected_agent = args.agents[0]

    row = summarize_eval_json(eval_json, agent_name=selected_agent)
    row["label"] = args.label
    summary_path = run_dir / "summary.csv"
    save_summary_csv([row], summary_path)

    print(f"\nCompleted feature run: {args.label}")
    print(f"Saved evaluation bundle: {run_dir}")
    print(f"Summary row: {row}")


if __name__ == "__main__":
    main()
