"""callbacks.py — q_agent v2: tabular Q-learning, full action set, D4 canon."""

import os
import pickle
import random

import numpy as np

from .features import state_to_features
from .symmetry import canonicalize_v2, IDENTITY

# --- Ablation switch (see week-2 experiment) --------------------------------
USE_SYMMETRY = os.environ.get("Q_AGENT_SYMMETRY", "1") != "0"


def canon(features):
    return canonicalize_v2(features) if USE_SYMMETRY else (features, IDENTITY)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "q_table_v2_sym.pkl" if USE_SYMMETRY else "q_table_v2_plain.pkl"


def setup(self):
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            self.q_table = pickle.load(f)
        self.logger.info(f"Loaded Q-table with {len(self.q_table)} states.")
    else:
        self.q_table = {}
        self.logger.info("No saved model found — empty Q-table.")
    # small epsilon even in play mode: breaks deterministic loops (see W1 bug)
    # play-mode exploration: masking already prevents loops and fatal picks,
    # so only a whisper of randomness is needed as tie-breaker insurance
    self.epsilon = float(os.environ.get("Q_AGENT_PLAY_EPS", "0.02")) \
        if not self.train else 1.0


def allowed_actions(features, game_state):
    """Mask out OBVIOUSLY fatal/impossible actions; learning chooses among
    the rest. Moves into blocked tiles, BOMB without escape (bomb_safe=0),
    and BOMB while unavailable are never offered — to exploration OR argmax."""
    allowed = [i for i in range(4) if features[1 + i] == 1]
    # WAIT is legal — except standing still inside a blast zone with a known
    # way out, which is certain death, so it is never offered then
    if not (features[5] == 3 and allowed):
        allowed.append(4)
    if not allowed:                                     # cornered: allow WAIT
        allowed.append(4)
    bomb_available = game_state['self'][2]
    if bomb_available and features[7] == 1:
        allowed.append(5)
    return allowed


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    allowed = allowed_actions(features, game_state)

    if random.random() < self.epsilon:
        return ACTIONS[random.choice(allowed)]

    canon_f, g = canon(features)
    q_values = self.q_table.get(canon_f, np.zeros(len(ACTIONS)))
    best_real = max(allowed, key=lambda a: q_values[g.apply_action(a)])
    self.logger.debug(f"{features} -> {canon_f} via {g} -> {ACTIONS[best_real]}")
    return ACTIONS[best_real]
