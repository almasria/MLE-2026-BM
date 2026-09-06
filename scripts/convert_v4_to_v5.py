#!/usr/bin/env python3
"""Warm-start a v5 Q-table from a trained v4 table.

Every v4 key becomes a v5 key with engagement = 0 (unknown in v4) and the
folded EMPTY bomb value merged into NONE. All solo knowledge carries over;
only the opponent-near states (engagement 1/2) start from zero, so a few
thousand rounds against opponents completes the table.

    python convert_v4_to_v5.py            # agent_code/q_agent/q_table_v4_sym.pkl -> v5
"""
import pickle
from pathlib import Path

src = Path("agent_code/q_agent/q_table_v4_sym.pkl")
dst = Path("agent_code/q_agent/q_table_v5_sym.pkl")
v4 = pickle.load(open(src, "rb"))
v5 = {}
for key, row in v4.items():
    key = list(key)
    if key[7] == 1:                 # BOMB_EMPTY -> BOMB_NONE
        key[7] = 0
    key = tuple(key) + (0,)         # engagement unknown -> ENGAGE_NONE
    v5[key] = (v5[key] + row) / 2 if key in v5 else row.copy()
pickle.dump(v5, open(dst, "wb"))
print(f"{len(v4)} v4 states -> {len(v5)} v5 states written to {dst}")
