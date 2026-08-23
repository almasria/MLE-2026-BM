#!/usr/bin/env python3
"""Audit a v2 Q-table against ground truth. Run inside agent_code/q_agent/."""
import pickle, sys
import numpy as np
qt = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "q_table_v2_sym.pkl", "rb"))
A = ['UP','RIGHT','DOWN','LEFT','WAIT','BOMB']
danger = [(f, v) for f, v in qt.items() if f[5] == 1]
bad_escape = [f for f, v in danger if f[6] != 4 and int(np.argmax(v)) != f[6]]
suicidal = [f for f, v in qt.items() if f[7] == 0 and int(np.argmax(v)) == 5]
blocked = [f for f, v in qt.items() if int(np.argmax(v)) < 4 and f[1 + int(np.argmax(v))] == 0]
print(f"states: {len(qt)} | danger states: {len(danger)}")
print(f"not following safe_dir: {len(bad_escape)} | suicidal-bomb argmax: {len(suicidal)} | into-blocked argmax: {len(blocked)}")
for f in bad_escape[:10]:
    print("  ", f, "-> argmax", A[int(np.argmax(qt[f]))], f"(safe_dir {A[f[6]]})")
