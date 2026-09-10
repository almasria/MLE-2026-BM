#!/usr/bin/env python3
"""Audit a trained Q-table against ground truth derived from its features.

    python scripts/audit_table.py agent_code/RL-Team/q_table_v5_sym.pkl

Reports escape discipline (danger states whose greedy action follows the
escape direction), suicidal or blocked greedy actions, and training coverage.
Works for v4 and v5 tables (indices 5, 6, 7 are shared).
"""
import pickle
import sys

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
F_URGENCY, F_SAFE_DIR, F_BOMB_OPPORTUNITY = 5, 6, 7

path = sys.argv[1] if len(sys.argv) > 1 else "agent_code/RL-Team/q_table_v5_sym.pkl"
qt = pickle.load(open(path, "rb"))

danger = [(f, v) for f, v in qt.items() if f[F_URGENCY] > 0]
bad_escape = [f for f, v in danger
              if f[F_SAFE_DIR] != 4 and int(np.argmax(v)) != f[F_SAFE_DIR]]
suicidal = [f for f, v in qt.items()
            if f[F_BOMB_OPPORTUNITY] == 0 and int(np.argmax(v)) == 5]
blocked = [f for f, v in qt.items()
           if int(np.argmax(v)) < 4 and f[1 + int(np.argmax(v))] == 0]
peaks = [np.abs(v).max() for v in qt.values()]
untrained = sum(1 for p in peaks if p < 1.0)
well = sum(1 for p in peaks if p > 10.0)

print(f"file: {path}")
print(f"states: {len(qt)} | tuple length: {len(next(iter(qt)))} | danger states: {len(danger)}")
if danger:
    ok = len(danger) - len(bad_escape)
    print(f"escape discipline: {ok}/{len(danger)} ({ok / len(danger) * 100:.0f}%)")
print(f"suicidal-bomb argmax: {len(suicidal)} | into-blocked argmax: {len(blocked)}")
print(f"coverage: {untrained} barely trained ({untrained / len(qt) * 100:.0f}%), "
      f"{well} well trained ({well / len(qt) * 100:.0f}%)")
for f in bad_escape[:10]:
    print("  ", f, "->", ACTIONS[int(np.argmax(qt[f]))], f"(safe_dir {ACTIONS[f[F_SAFE_DIR]]})")
