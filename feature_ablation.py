"""
feature_ablation.py

A lightweight experimental runner for feature comparisons.

This script selects a committed feature implementation through an environment
variable. It never copies files over the tracked feature dispatcher.

Use it like this:

    python3 feature_ablation.py --label strategic --feature-version v3 \
        --rounds 1000 --eval-rounds 20

Each run writes its Q-model and training log directly into its own output
directory, so tracked agent artifacts are never reset or overwritten.

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
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from agent_code.q_agent.config import GAME_REWARDS, q_learning_manifest

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "agent_code" / "q_agent"
OUT_ROOT = ROOT / "evaluation_results"
FEATURE_FILES = {"v3": AGENT_DIR / "featuresv3.py"}


def git_output(*args):
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def relevant_git_changes():
    """Return changed paths, excluding personal editor metadata."""
    changes = []
    for line in git_output("status", "--porcelain").splitlines():
        path = line[3:]
        if path == ".vscode" or path.startswith(".vscode/"):
            continue
        changes.append(line)
    return changes


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest, path: Path):
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


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
        "score_per_round": score / rounds if rounds else 0.0,
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
        "score_per_round",
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
    ap.add_argument(
        "--feature-version",
        default="v3",
        choices=["v3"],
        help="Compatible feature implementation to use for this run",
    )
    ap.add_argument("--agents", nargs="+", default=["q_agent"], help="Agents to deploy, e.g. --agents q_agent or --agents q_agent rule_based_agent")
    ap.add_argument("--train", type=int, default=1, help="How many of the first agents are in training mode")
    ap.add_argument("--rounds", type=int, default=1000, help="Training rounds")
    ap.add_argument("--eval-rounds", type=int, default=20, help="Evaluation rounds")
    ap.add_argument("--scenario", default="coin-heaven", choices=["empty", "coin-heaven", "loot-crate", "classic"])
    ap.add_argument("--seed", type=int, default=0, help="Seed for this run")
    ap.add_argument("--symmetry", type=int, default=1, choices=[0, 1], help="1 for D4 symmetry on, 0 off")
    ap.add_argument("--no-gui", action="store_true", help="run without GUI")
    args = ap.parse_args()

    created_at = datetime.now().astimezone()
    stamp = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUT_ROOT / stamp / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["Q_AGENT_SYMMETRY"] = str(args.symmetry)
    env["Q_AGENT_SEED"] = str(args.seed)
    env["Q_AGENT_RUN_TAG"] = f"{args.label}_seed{args.seed}"
    env["Q_AGENT_FEATURE_VERSION"] = args.feature_version
    model_name = (
        "q_table_v3_sym.pkl" if args.symmetry else "q_table_v3_plain.pkl"
    )
    env["Q_AGENT_MODEL_PATH"] = str((run_dir / model_name).resolve())
    env["Q_AGENT_HISTORY_PATH"] = str(
        (run_dir / "training_history.csv").resolve()
    )

    feature_path = FEATURE_FILES[args.feature_version]
    changed_paths = relevant_git_changes()
    manifest = {
        "label": args.label,
        "created_at": created_at.isoformat(),
        "status": "running",
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(changed_paths),
            "changes": changed_paths,
        },
        "feature": {
            "version": args.feature_version,
            "file": str(feature_path.relative_to(ROOT)),
            "sha256": file_sha256(feature_path),
        },
        "training": {
            "seed": args.seed,
            "scenario": args.scenario,
            "agents": args.agents,
            "train_agents": args.train,
            "rounds": args.rounds,
        },
        "evaluation": {
            "seed": args.seed,
            "scenario": args.scenario,
            "agents": args.agents,
            "rounds": args.eval_rounds,
        },
        "q_learning": {
            **q_learning_manifest(),
            "symmetry": bool(args.symmetry),
        },
        "rewards": dict(sorted(GAME_REWARDS.items())),
        "artifacts": {
            "model": model_name,
            "training_history": "training_history.csv",
            "evaluation": f"eval_{args.label}.json",
            "summary": "summary.csv",
        },
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

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
    try:
        run_cmd(train_cmd, env=env)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_manifest(manifest, manifest_path)
        raise

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
    try:
        run_cmd(eval_cmd, env=env)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_manifest(manifest, manifest_path)
        raise

    selected_agent = args.agents[0] if args.agents else None
    if args.train > 0 and args.agents:
        selected_agent = args.agents[0]

    row = summarize_eval_json(eval_json, agent_name=selected_agent)
    row["label"] = args.label
    summary_path = run_dir / "summary.csv"
    save_summary_csv([row], summary_path)

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    write_manifest(manifest, manifest_path)

    print(f"\nCompleted feature run: {args.label}")
    print(f"Saved evaluation bundle: {run_dir}")
    print(f"Summary row: {row}")


if __name__ == "__main__":
    main()
