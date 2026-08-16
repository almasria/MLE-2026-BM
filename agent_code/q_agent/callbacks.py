"""
callbacks.py — Stage 1 coin-collector agent for ukoethe/bomberman_rl.

Tabular Q-learning over a tiny engineered feature space:
    feature = (direction_to_nearest_coin, up_free, right_free, down_free, left_free)

That's it. With good features, this small state space is enough to learn
near-perfect coin collection. Extend `state_to_features` for later stages
(crates, danger, escape routes) — the rest of the pipeline stays the same.

Framework contract (called by the environment):
    setup(self)            -> called once before a set of games
    act(self, game_state)  -> must return one of ACTIONS
"""

import os
import pickle
import random
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Stage 1: coins only — we never need BOMB, and WAIT is rarely useful.
# Restricting the action set makes learning much faster. Widen it in stage 2.
STAGE1_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

MODEL_FILE = "q_table.pkl"

# Direction encoding for the "where is the nearest coin" feature
# 0 = up, 1 = right, 2 = down, 3 = left, 4 = no coin reachable / on top of coin
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # matches UP, RIGHT, DOWN, LEFT


def setup(self):
    """Called once before a set of games. Load the Q-table if it exists."""
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            self.q_table = pickle.load(f)
        self.logger.info(f"Loaded Q-table with {len(self.q_table)} states.")
    else:
        self.q_table = {}
        self.logger.info("No saved model found — starting with an empty Q-table.")

    # Exploration rate. train.py decays this over rounds; when just playing
    # (not training), we keep a tiny epsilon so behavior is near-greedy.
    self.epsilon = 0.1 if not self.train else 1.0


def act(self, game_state: dict) -> str:
    """Choose an action with an epsilon-greedy policy over the Q-table."""
    features = state_to_features(game_state)

    # Exploration. NOTE: we keep a small epsilon even in play mode (not just
    # training) — a fully deterministic policy in a deterministic world can get
    # trapped bouncing between two states forever. A little noise breaks loops.
    if random.random() < self.epsilon:
        return random.choice(STAGE1_ACTIONS)

    # Exploitation: pick argmax over Q-values (unknown states default to zeros)
    q_values = self.q_table.get(features, np.zeros(len(STAGE1_ACTIONS)))
    best = int(np.argmax(q_values))
    self.logger.debug(f"Features {features} -> Q {np.round(q_values, 2)} -> {STAGE1_ACTIONS[best]}")
    return STAGE1_ACTIONS[best]


# ---------------------------------------------------------------------------
# Feature engineering — the heart of the agent
# ---------------------------------------------------------------------------

def state_to_features(game_state: dict):
    """
    Compress the full game_state dict into a small, hashable feature tuple.

    Returns
    -------
    tuple: (coin_direction, up_free, right_free, down_free, left_free)
        coin_direction in {0: up, 1: right, 2: down, 3: left, 4: none}
        *_free in {0, 1} — whether that neighboring tile is walkable
    """
    if game_state is None:
        return None

    field = game_state['field']          # -1 wall, 0 free, 1 crate
    x, y = game_state['self'][3]         # our position
    coins = game_state['coins']          # list of (x, y)

    coin_direction = bfs_direction_to_nearest(field, (x, y), set(coins))

    neighbors = []
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        neighbors.append(1 if field[nx, ny] == 0 else 0)

    return (coin_direction, *neighbors)


def bfs_direction_to_nearest(field, start, targets) -> int:
    """
    Breadth-first search from `start` over free tiles (field == 0).
    Returns the index of the FIRST STEP direction (0..3) on a shortest path
    to the nearest target, or 4 if no target is reachable (or we're on one).

    This one helper generalizes to later stages: pass crates, safe tiles,
    or opponents as `targets` to get "direction to nearest X" features.
    """
    if not targets or start in targets:
        return 4

    # Standard BFS, remembering the first move that led to each visited tile
    queue = deque([start])
    first_move = {start: None}

    while queue:
        current = queue.popleft()
        if current in targets:
            return first_move[current]

        cx, cy = current
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (cx + dx, cy + dy)
            if nxt in first_move:
                continue
            if field[nxt[0], nxt[1]] != 0:   # wall or crate blocks the path
                continue
            # Propagate the first move: if we're one step from start, this
            # direction IS the first move; otherwise inherit it.
            first_move[nxt] = i if first_move[current] is None else first_move[current]
            queue.append(nxt)

    return 4  # no coin reachable
