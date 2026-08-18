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

import numpy as np

from .features import state_to_features
from .symmetry import canonicalize_v1

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

    # Exploitation with D4 canonicalization: Q-values live only at the
    # orbit representative. Pick the greedy action THERE, then map it back
    # into the real world through the inverse group element.
    canon, g = canonicalize_v1(features)
    q_values = self.q_table.get(canon, np.zeros(len(STAGE1_ACTIONS)))
    best_canon = int(np.argmax(q_values))
    best_real = g.inverse().apply_action(best_canon)
    self.logger.debug(f"{features} -> canon {canon} via {g} -> {STAGE1_ACTIONS[best_real]}")
    return STAGE1_ACTIONS[best_real]
