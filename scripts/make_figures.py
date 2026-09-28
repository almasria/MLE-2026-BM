#!/usr/bin/env python3
"""Figures for the report, built from the training history and stage evaluations.

    python scripts/make_figures.py
    python scripts/make_figures.py --history results/training_history.csv \
        --evals "results/eval_*.json" --out figures

Writes PDF (for LaTeX) and PNG (for previews) into --out:

  symmetry_ablation   rolling mean of coins per training round on coin-heaven,
                      canonicalization on vs off, one thin line per run and the
                      condition mean on top; prints rounds-to-45 per run
  curriculum          one training run through the curriculum stages: coins per
                      round, table size and exploration rate, stage by stage
  stage_evals         score of the agent and of the best rule-based opponent in
                      the evaluation that follows each curriculum stage

Runs are separated by the round counter restarting, not by run tag: the same
tag can occur in several runs (the symmetry ablation was run twice with
identical tags), and grouping by tag would interleave them.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# history columns written by train.end_of_round
COLS = ("tag", "symmetry", "round", "reward", "coins", "epsilon", "table_size")

WINDOW = 100          # rolling-mean window, in rounds
THRESHOLD = 45        # coins per round that counts as converged on coin-heaven

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
NEUTRAL = "#a3a29c"   # opponents, discarded stages
SURFACE = "#ffffff"

STAGE_NAMES = {
    "1": "coin-heaven", "2": "loot-crate", "3": "vs peaceful",
    "4": "vs coin collector", "5": "vs rule-based", "6": "self-play",
}

# two-line tick labels for the stage-evaluation bars
STAGE_LABELS = {
    "1": "coin-heaven", "2": "loot-crate", "3": "vs peaceful",
    "4": "vs coin\ncollector", "5": "vs rule-\nbased", "6": "self-play",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


# --- data ---------------------------------------------------------------------

def load_runs(path):
    """List of runs; each run is a dict of column arrays plus its tag."""
    runs, current, last_round = [], None, None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) != len(COLS):
                continue
            try:
                rnd = int(row[2])
            except ValueError:
                continue                          # header or corrupt line
            tag = row[0]
            if current is None or tag != current["tag"] or rnd <= last_round:
                current = {"tag": tag, "symmetry": int(row[1]), "rows": []}
                runs.append(current)
            current["rows"].append(row)
            last_round = rnd
    out = []
    for r in runs:
        a = np.array(r["rows"])
        out.append({
            "tag": r["tag"], "symmetry": r["symmetry"],
            "round": a[:, 2].astype(int), "reward": a[:, 3].astype(float),
            "coins": a[:, 4].astype(float), "epsilon": a[:, 5].astype(float),
            "table_size": a[:, 6].astype(int),
        })
    return out


def rolling(x, w=WINDOW):
    """Rolling mean with a growing window over the first w-1 rounds."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def rounds_to(x, threshold=THRESHOLD, w=WINDOW):
    """First round whose full-window rolling mean reaches the threshold."""
    x = np.asarray(x, float)
    if len(x) < w:
        return None
    m = np.convolve(x, np.ones(w) / w, mode="valid")
    hit = np.nonzero(m >= threshold)[0]
    return int(hit[0] + w) if len(hit) else None


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf / .png")


# --- figure 1: symmetry ablation --------------------------------------------

def fig_symmetry(runs, out):
    abl = [r for r in runs if r["tag"].startswith("sym")]
    if not abl:
        print("  no symmetry-ablation runs found; skipped")
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    summary = {}
    for sym, color, label in ((1, SERIES_1, "canonicalization on"),
                              (0, SERIES_2, "canonicalization off")):
        group = [r for r in abl if r["symmetry"] == sym]
        n = min(len(r["coins"]) for r in group)
        curves = np.array([rolling(r["coins"][:n]) for r in group])
        x = np.arange(1, n + 1)
        for c in curves:
            ax.plot(x, c, color=color, lw=0.8, alpha=0.3)
        mean = curves.mean(axis=0)
        ax.plot(x, mean, color=color, lw=2.0, label=label)
        conv = [rounds_to(r["coins"]) for r in group]
        summary[label] = (conv, [int(r["table_size"][-1]) for r in group])
    ax.axhline(THRESHOLD, color=INK_2, lw=1.0, ls="--")
    ax.annotate(f"{THRESHOLD} coins", (0, THRESHOLD), xytext=(4, 3),
                textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xlabel("training round")
    ax.set_ylabel(f"coins per round ({WINDOW}-round mean)")
    ax.set_ylim(0, 52)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "symmetry_ablation")

    print(f"  rounds to a {WINDOW}-round mean of {THRESHOLD} coins, per run:")
    for label, (conv, tables) in summary.items():
        vals = [v for v in conv if v is not None]
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        print(f"    {label}: {conv}  mean {np.mean(vals):.0f}, sd {sd:.0f}, "
              f"n={len(vals)}; final table sizes {sorted(set(tables))}")


# --- figure 2: curriculum training run --------------------------------------

def fig_curriculum(runs, out):
    stages = [r for r in runs if re.match(r"stage\d", r["tag"])]
    if not stages:
        print("  no stage-tagged runs found; skipped")
        return
    # keep the last complete pass through the stages
    starts = [i for i, r in enumerate(stages) if r["tag"].startswith("stage1")]
    stages = stages[starts[-1]:] if starts else stages

    fig, axes = plt.subplots(3, 1, figsize=(6.3, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]})
    offset = 0
    for r in stages:
        num = re.match(r"stage(\d)", r["tag"]).group(1)
        name = STAGE_NAMES.get(num, r["tag"])
        discarded = name == "self-play"
        color = NEUTRAL if discarded else SERIES_1
        n = len(r["coins"])
        x = np.arange(offset + 1, offset + n + 1)
        axes[0].plot(x, rolling(r["coins"]), color=color, lw=1.5)
        axes[1].plot(x, r["table_size"], color=color, lw=1.5)
        axes[2].plot(x, r["epsilon"], color=color, lw=1.5)
        for ax in axes:
            ax.axvline(offset, color=INK_2, lw=0.6)
            if discarded:
                ax.axvspan(offset, offset + n, color=GRID, alpha=0.6, lw=0)
        axes[0].text(offset + n / 2, 1.02, num, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8, color=INK_2)
        offset += n
    axes[0].set_ylabel(f"coins per round\n({WINDOW}-round mean)")
    axes[1].set_ylabel("table entries")
    axes[2].set_ylabel("exploration $\\varepsilon$")
    axes[2].set_xlabel("training round (cumulative over stages)")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlim(0, offset)
    save(fig, out, "curriculum")
    print("  stages: " + ", ".join(f"{r['tag']} ({len(r['coins'])} rounds)" for r in stages))
    print("  stage numbers above the top panel; name them in the caption")


# --- figure 3: evaluation after each stage -----------------------------------

def eval_date(by_round):
    """Date of the first round in a stats file, from keys like
    'Round 01 (2026-09-12 19-59-23)'."""
    for key in by_round:
        m = re.search(r"\((\d{4}-\d{2}-\d{2})", key)
        if m:
            return datetime.date.fromisoformat(m.group(1))
    return None


def fig_stage_evals(pattern, out, include_all=False,
                    agent_prefixes=("RL-Team", "q_agent")):
    """Bars per curriculum stage. Only files named eval_<n>-<n>-<name>.json
    count as stage evaluations; others (eval_econ.json, ...) are skipped.
    Stage files from an older pipeline run (more than two days before the
    newest one) are left out unless include_all is set, since the pipeline
    overwrites eval_1..5 but leaves a stale eval_6 behind."""
    found = []
    for f in sorted(glob.glob(pattern)):
        m = re.fullmatch(r"eval_(\d)-\d-([a-z-]+)\.json", os.path.basename(f))
        if not m:
            print(f"  {f}: not a stage evaluation; skipped")
            continue
        raw = json.load(open(f))
        found.append((m.group(1), f, raw, eval_date(raw.get("by_round", {}))))
    if not found:
        print(f"  no stage evaluation files match {pattern!r}; skipped")
        return

    dates = [d for *_, d in found if d]
    newest = max(dates) if dates else None
    labels, ours, best, rounds = [], [], [], []
    for num, f, raw, date in found:
        if (not include_all and newest and date
                and (newest - date).days > 2):
            print(f"  {f}: from {date}, an older run than {newest}; skipped "
                  f"(--all-evals to include)")
            continue
        data = raw["by_agent"]
        me = next((v for k, v in data.items() if k.startswith(agent_prefixes)), None)
        if me is None:
            print(f"  {f}: agent not found; skipped")
            continue
        opp = [v for k, v in data.items() if not k.startswith(agent_prefixes)]
        n = me.get("rounds", 1)
        labels.append(f"{num}\n{STAGE_LABELS.get(num, num)}")
        ours.append(me.get("score", 0) / n)
        best.append(max(o.get("score", 0) for o in opp) / n if opp else 0)
        rounds.append(n)
        print(f"  {f}: {date}, {n} rounds")

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    b1 = ax.bar(x - w / 2 - 0.01, ours, w, color=SERIES_1, label="our agent")
    b2 = ax.bar(x + w / 2 + 0.01, best, w, color=NEUTRAL, label="best rule-based opponent")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("score per round")
    ax.set_xlabel("evaluation after curriculum stage")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_ylim(0, max(ours + best) * 1.15)
    save(fig, out, "stage_evals")
    print("  per stage (score/round ours vs best opponent, rounds): " +
          "; ".join(f"{l.replace(chr(10), ' ')}: {o:.2f} vs {b:.2f} ({n})"
                    for l, o, b, n in zip(labels, ours, best, rounds)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", default="results/training_history.csv")
    ap.add_argument("--evals", default="results/eval_*.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--all-evals", action="store_true",
                    help="include stage evaluations from older pipeline runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(args.history):
        runs = load_runs(args.history)
        print(f"{args.history}: {len(runs)} runs")
        fig_symmetry(runs, out)
        fig_curriculum(runs, out)
    else:
        print(f"{args.history} not found; training figures skipped")
    fig_stage_evals(args.evals, out, include_all=args.all_evals)


if __name__ == "__main__":
    main()
