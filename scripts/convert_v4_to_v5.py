#!/usr/bin/env python3
"""Warm-start a v5 Q-table from a trained v4 table.

Each v4 key becomes a v5 key with engagement = 0; the v4 "empty bomb" value
is folded into "none". Opponent-related states start from zero and need the
opponent training stages.

    python scripts/convert_v4_to_v5.py
"""
import pickle
from pathlib import Path

src = Path("agent_code/RL-Team/q_table_v4_sym.pkl")
dst = Path("agent_code/RL-Team/q_table_v5_sym.pkl")
v4 = pickle.load(open(src, "rb"))
v5 = {}
for key, row in v4.items():
    key = list(key)
    if key[7] == 1:
        key[7] = 0
    key = tuple(key) + (0,)
    v5[key] = (v5[key] + row) / 2 if key in v5 else row.copy()
pickle.dump(v5, open(dst, "wb"))
print(f"{len(v4)} v4 states -> {len(v5)} v5 states written to {dst}")
