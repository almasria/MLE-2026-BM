#!/usr/bin/env python3
"""
run_sweep.py — fair hyperparameter sweeps for q_agent.

For EACH value x EACH seed the agent trains FRESH through the full curriculum
(so the swept parameter is the only difference between runs), then plays 100
evaluation rounds against three rule_based agents. Every trained table is
archived and one summary row is written to results/sweep_summary.csv.

Safety properties of this version:
  * NEVER touches your real table. Each run trains into a scratch file via
    the agent's Q_AGENT_MODEL_PATH override (agent_code/q_agent/sweep_*.pkl),
    so q_table_v5_sym.pkl (the tournament model) is left alone.
  * Every stage has a time limit (--stage-timeout, default 3 h). A stage that
    exceeds it is killed, the run is recorded as TIMEOUT, and the sweep
    continues with the next run instead of hanging forever.
  * Resumable: (value, seed) pairs already in the summary CSV are skipped, so
    a killed sweep restarts only the unfinished work.
  * Progress is visible: each stage streams to results/sweep_logs/<run>_<stage>.log
    and prints its elapsed time when done. If a stage's log stops growing for
    a long time, that stage is stuck.

Usage (repo root):
    python run_sweep.py --param Q_AGENT_GAMMA --values 0.9 0.95 --seeds 2
    python run_sweep.py --param Q_AGENT_ALPHA --values 0.05 0.1 0.2 --seeds 2
    python run_sweep.py --param Q_AGENT_GAMMA --values 0.9 --seeds 1 --scale 0.02   # 1-min plumbing check

Read results in results/sweep_summary.csv; compare the MEAN score across
seeds per value (printed at the end), never a single run.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

AGENT_DEFAULT = "q_agent"

# (scenario, opponents, rounds, eps_start or None for fresh 1.0) -- ends on the
# tournament benchmark opponent, like train_tournament.py
CURRICULUM = [
    ("coin-heaven", [], 400, None),
    ("loot-crate", [], 3000, "0.3"),
    ("classic", ["peaceful_agent"] * 3, 1500, "0.2"),
    ("classic", ["coin_collector_agent"] * 3, 3000, "0.1"),
    ("classic", ["rule_based_agent"] * 3, 3000, "0.05"),
]
EVAL_OPPONENTS = ["rule_based_agent"] * 3
EVAL_ROUNDS = 100

SUMMARY = Path("results/sweep_summary.csv")
ARCHIVE = Path("results/sweep_models")
LOGS = Path("results/sweep_logs")
HEADER = "param,value,seed,status,score,best_opp,coins,kills,suicides,table_size\n"


def run_stage(cmd, env, log_path, timeout_s):
    """Run one main.py invocation, streaming output to a log file.
    Returns 'ok', 'timeout' or 'error'."""
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "ok" if proc.returncode == 0 else "error"


def already_done(param):
    done = set()
    if SUMMARY.exists():
        for row in csv.DictReader(open(SUMMARY)):
            if row["param"] == param and row["status"] == "ok":
                done.add((row["value"], row["seed"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=AGENT_DEFAULT)
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all round counts (0.02 = quick plumbing check)")
    ap.add_argument("--stage-timeout", type=float, default=3.0,
                    help="hours allowed per curriculum stage before it is killed")
    args = ap.parse_args()

    agent_dir = Path("agent_code") / args.agent
    for d in (Path("results"), ARCHIVE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    if not SUMMARY.exists():
        SUMMARY.write_text(HEADER)
    done = already_done(args.param)
    timeout_s = args.stage_timeout * 3600

    for value in args.values:
        for seed in range(args.seeds):
            if (str(value), str(seed)) in done:
                print(f"skip {args.param}={value} seed {seed} (already in summary)")
                continue

            tag = f"{args.param}={value}_s{seed}"
            scratch_name = f"sweep_{tag}.pkl".replace("=", "_")
            scratch = agent_dir / scratch_name
            if scratch.exists():
                scratch.unlink()                      # fresh start for THIS run only

            # the agent writes/reads its table relative to its own folder, so the
            # override is just the file name; the real q_table_v5_sym.pkl is untouched
            env = dict(os.environ,
                       Q_AGENT_SEED=str(seed),
                       Q_AGENT_RUN_TAG=tag,
                       Q_AGENT_MODEL_PATH=scratch_name,
                       **{args.param: str(value)})

            print(f"\n=== {tag} : fresh curriculum "
                  f"({time.strftime('%H:%M:%S')}) ===", flush=True)
            status = "ok"
            for scenario, opponents, rounds, eps in CURRICULUM:
                n = max(1, int(rounds * args.scale))
                stage_env = dict(env)
                if eps is not None:
                    stage_env["Q_AGENT_EPS_START"] = eps
                stage_tag = f"{scenario}-{opponents[0] if opponents else 'solo'}"
                t0 = time.time()
                status = run_stage(
                    [sys.executable, "main.py", "play", "--agents", args.agent,
                     *opponents, "--train", "1", "--scenario", scenario,
                     "--n-rounds", str(n), "--no-gui"],
                    stage_env, LOGS / f"{tag}_{stage_tag}.log", timeout_s)
                print(f"    {stage_tag:32s} {n:5d} rounds  "
                      f"{(time.time() - t0) / 60:6.1f} min  [{status}]", flush=True)
                if status != "ok":
                    break

            row = dict(score=0, best_opp=0, coins=0, kills=0, suicides=0, size=0)
            if status == "ok":
                n_eval = max(5, int(EVAL_ROUNDS * args.scale))
                stats = f"results/{tag}_eval.json"
                t0 = time.time()
                status = run_stage(
                    [sys.executable, "main.py", "play", "--agents", args.agent,
                     *EVAL_OPPONENTS, "--scenario", "classic", "--n-rounds",
                     str(n_eval), "--no-gui", "--save-stats", stats],
                    env, LOGS / f"{tag}_eval.log", timeout_s)
                print(f"    {'evaluation vs rule_based':32s} {n_eval:5d} rounds  "
                      f"{(time.time() - t0) / 60:6.1f} min  [{status}]", flush=True)
                if status == "ok":
                    data = json.load(open(stats))["by_agent"]
                    me = data.get(args.agent, {})
                    opp = [v.get("score", 0) for k, v in data.items() if k != args.agent]
                    import pickle
                    row.update(score=me.get("score", 0), best_opp=max(opp) if opp else 0,
                               coins=me.get("coins", 0), kills=me.get("kills", 0),
                               suicides=me.get("suicides", 0),
                               size=len(pickle.load(open(scratch, "rb"))) if scratch.exists() else 0)
                    if scratch.exists():
                        shutil.copy(scratch, ARCHIVE / f"{tag}.pkl")

            with open(SUMMARY, "a") as fsum:
                fsum.write(f"{args.param},{value},{seed},{status},{row['score']},"
                           f"{row['best_opp']},{row['coins']},{row['kills']},"
                           f"{row['suicides']},{row['size']}\n")
            print(f"  -> {status}: score {row['score']} vs best opp {row['best_opp']} | "
                  f"kills {row['kills']} | suicides {row['suicides']} | "
                  f"{row['size']} states", flush=True)

    print("\n=== summary (mean score across completed seeds) ===")
    by_val = defaultdict(list)
    for r in csv.DictReader(open(SUMMARY)):
        if r["param"] == args.param and r["status"] == "ok":
            by_val[r["value"]].append(int(r["score"]))
    for value in args.values:
        scores = by_val.get(str(value), [])
        if scores:
            print(f"  {args.param}={value}: mean {sum(scores) / len(scores):.0f} "
                  f"over {len(scores)} seed(s)  {scores}")
        else:
            print(f"  {args.param}={value}: no completed runs")


if __name__ == "__main__":
    main()
