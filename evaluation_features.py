"""
Standalone feature-evaluation report generator.

This script does NOT modify analyze_symmetry.py.
It reads the existing training and evaluation artifacts, then saves a
self-contained bundle under:

    evaluation_results/<timestamp>/

The bundle includes:
- summary.csv with rows: run_tag, symmetry, seed, avg_reward, avg_coins,
  avg_steps, total_score, kills, suicides, table_size
- plots for avg_coins, avg_reward, avg_steps vs feature variant / condition
- raw copied evaluation JSON files
- a short report.txt explaining what was generated

Usage:
    python evaluation_features.py
"""

import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_HISTORY = ROOT / "agent_code" / "q_agent" / "training_history.csv"
SUMMARY_CSV = ROOT / "results" / "ablation_summary.csv"
RESULTS_DIR = ROOT / "results"
EVAL_ROOT = ROOT / "evaluation_results"


def load_training_history(path: Path):
    """Load training history rows and return a dictionary keyed by run_tag."""
    rows = []
    if not path.exists():
        return rows

    with open(path, newline="") as f:
        reader = csv.DictReader(f, fieldnames=["run_tag", "symmetry", "round", "reward", "coins", "epsilon", "table_size"])
        for row in reader:
            rows.append(row)
    return rows


def summarize_eval_jsons(eval_dir: Path):
    """Read all *_eval.json files in results and summarize per run."""
    summary = []
    for fp in sorted(eval_dir.glob("*_eval.json")):
        with open(fp) as f:
            data = json.load(f)

        agent_name = next(iter(data["by_agent"].keys()))
        agent_stats = data["by_agent"][agent_name]
        round_stats = data.get("by_round", {})

        coins = float(agent_stats.get("coins", 0))
        score = float(agent_stats.get("score", 0))
        steps = float(agent_stats.get("steps", 0))
        rounds = float(agent_stats.get("rounds", 0))
        kills = float(sum(r.get("kills", 0) for r in round_stats.values()))
        suicides = float(sum(r.get("suicides", 0) for r in round_stats.values()))

        avg_reward = score / rounds if rounds else 0.0
        avg_coins = coins / rounds if rounds else 0.0
        avg_steps = steps / rounds if rounds else 0.0

        summary.append({
            "run_tag": fp.stem.replace("_eval", ""),
            "seed": fp.stem.split("seed")[-1] if "seed" in fp.stem else "unknown",
            "symmetry": 1 if "sym1" in fp.stem else 0,
            "avg_reward": avg_reward,
            "avg_coins": avg_coins,
            "avg_steps": avg_steps,
            "total_score": score,
            "kills": kills,
            "suicides": suicides,
            "table_size": None,
        })
    return summary


def summarize_ablation_csv(path: Path):
    """Add table_size when ablation_summary.csv is available."""
    if not path.exists():
        return {}

    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["run_tag"]] = int(row["table_size"])
    return out


def rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([])
    out = np.empty(len(values), dtype=float)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def make_plots(rows, bundle_dir: Path):
    """Create performance comparison plots."""
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["symmetry"]].append(r)

    # avg_coins plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for sym, rs in sorted(by_condition.items()):
        xs = [r["run_tag"] for r in rs]
        ys = [r["avg_coins"] for r in rs]
        ax.bar(xs, ys, alpha=0.7, label=f"symmetry={sym}")
    ax.set_title("Average coins per round by condition")
    ax.set_ylabel("avg_coins")
    ax.set_xlabel("run_tag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(bundle_dir / "avg_coins_by_condition.png", dpi=150)
    plt.close(fig)

    # avg_reward plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for sym, rs in sorted(by_condition.items()):
        xs = [r["run_tag"] for r in rs]
        ys = [r["avg_reward"] for r in rs]
        ax.bar(xs, ys, alpha=0.7, label=f"symmetry={sym}")
    ax.set_title("Average reward per round by condition")
    ax.set_ylabel("avg_reward")
    ax.set_xlabel("run_tag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(bundle_dir / "avg_reward_by_condition.png", dpi=150)
    plt.close(fig)

    # avg_steps plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for sym, rs in sorted(by_condition.items()):
        xs = [r["run_tag"] for r in rs]
        ys = [r["avg_steps"] for r in rs]
        ax.bar(xs, ys, alpha=0.7, label=f"symmetry={sym}")
    ax.set_title("Average steps per round by condition")
    ax.set_ylabel("avg_steps")
    ax.set_xlabel("run_tag")
    ax.legend()
    fig.tight_layout()
    fig.savefig(bundle_dir / "avg_steps_by_condition.png", dpi=150)
    plt.close(fig)

    # training curve if training history exists
    training_rows = load_training_history(TRAIN_HISTORY)
    if training_rows:
        per_run = defaultdict(list)
        for row in training_rows:
            per_run[row["run_tag"]].append(float(row["coins"]))

        fig, ax = plt.subplots(figsize=(10, 5))
        for run_tag, values in sorted(per_run.items()):
            sym = 1 if "sym1" in run_tag else 0
            color = "tab:blue" if sym == 1 else "tab:orange"
            roll = rolling_mean(values, 100)
            ax.plot(np.arange(1, len(values) + 1), roll, label=f"{run_tag}", color=color, alpha=0.8)
        ax.set_title("Training coin curves (rolling mean, window=100)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Coins per round (rolling average)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(bundle_dir / "training_coin_curves.png", dpi=150)
        plt.close(fig)


def write_summary_csv(rows, bundle_dir: Path):
    fieldnames = [
        "run_tag",
        "symmetry",
        "seed",
        "avg_reward",
        "avg_coins",
        "avg_steps",
        "total_score",
        "kills",
        "suicides",
        "table_size",
    ]
    with open(bundle_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bundle_dir = EVAL_ROOT / stamp
    bundle_dir.mkdir(parents=True, exist_ok=True)
    raw_eval_dir = bundle_dir / "raw_eval"
    raw_eval_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = summarize_eval_jsons(RESULTS_DIR)
    table_sizes = summarize_ablation_csv(SUMMARY_CSV)
    for row in eval_rows:
        key = row["run_tag"]
        if key in table_sizes:
            row["table_size"] = table_sizes[key]

    # ensure raw eval jsons are preserved
    for fp in sorted(RESULTS_DIR.glob("*_eval.json")):
        shutil.copy2(fp, raw_eval_dir / fp.name)

    write_summary_csv(eval_rows, bundle_dir)
    make_plots(eval_rows, bundle_dir)

    report = [
        "Evaluation feature summary",
        "========================",
        f"Timestamp: {stamp}",
        f"Training history file: {TRAIN_HISTORY}",
        f"Eval JSON directory: {RESULTS_DIR}",
        f"Bundle directory: {bundle_dir}",
        "",
        "Included outputs:",
        "- summary.csv",
        "- avg_coins_by_condition.png",
        "- avg_reward_by_condition.png",
        "- avg_steps_by_condition.png",
        "- training_coin_curves.png",
        "- raw_eval/*.json",
        "",
        "Columns in summary.csv:",
        "run_tag, symmetry, seed, avg_reward, avg_coins, avg_steps, total_score, kills, suicides, table_size",
    ]
    (bundle_dir / "report.txt").write_text("\n".join(report) + "\n")

    print(f"Evaluation bundle saved to: {bundle_dir}")
    print(f"Summary CSV: {bundle_dir / 'summary.csv'}")
    print(f"All plots: {bundle_dir}")


if __name__ == "__main__":
    main()
