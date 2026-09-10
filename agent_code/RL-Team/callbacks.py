"""RL-Team: tabular Q-learning with D4 canonicalisation and a safety mask.

Framework entry points: setup(self) once per session, act(self, game_state)
once per step.
"""

import os
import pickle
import random

import numpy as np

from .features import (BOMB_EMPTY, BOMB_NONE, F_BOMB_OPPORTUNITY, F_URGENCY,
                       MODEL_BASENAME, URGENCY_IMMINENT, canonicalize,
                       state_to_features, survivable_actions)
from .symmetry import IDENTITY

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Q_AGENT_SYMMETRY=0 disables canonicalisation (ablation); the code path is
# unchanged because the identity element maps every action to itself.
USE_SYMMETRY = os.environ.get("Q_AGENT_SYMMETRY", "1") != "0"
# empty bombs (survivable, hit nothing) are not offered unless enabled
ALLOW_EMPTY_BOMB = os.environ.get("Q_AGENT_ALLOW_EMPTY_BOMB", "0") == "1"
# Q_AGENT_DEEP_PRUNING=0 keeps only the one-step mask (ablation)
DEEP_PRUNING = os.environ.get("Q_AGENT_DEEP_PRUNING", "1") != "0"

DEFAULT_MODEL_FILE = f"{MODEL_BASENAME}_{'sym' if USE_SYMMETRY else 'plain'}.pkl"
MODEL_FILE = os.environ.get("Q_AGENT_MODEL_PATH", DEFAULT_MODEL_FILE)


def canon(features):
    return canonicalize(features) if USE_SYMMETRY else (features, IDENTITY)


def setup(self):
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            self.q_table = pickle.load(f)
        self.logger.info(f"Loaded Q-table with {len(self.q_table)} states.")
    else:
        self.q_table = {}
        self.logger.info("No saved model found; starting with an empty Q-table.")
    # a little randomness in play mode breaks deterministic two-state loops
    self.epsilon = (float(os.environ.get("Q_AGENT_PLAY_EPS", "0.02"))
                    if not self.train else 1.0)


def allowed_actions(features, game_state):
    """Actions that are not obviously fatal or impossible.

    Moves into blocked tiles and bombs without an escape are never offered,
    neither to exploration nor to the greedy choice. The learned policy
    chooses among what remains.
    """
    allowed = [i for i in range(4) if features[1 + i] == 1]
    # WAIT is withheld when a blast is imminent and a move exists
    if not (features[F_URGENCY] == URGENCY_IMMINENT and allowed):
        allowed.append(4)
    if not allowed:
        allowed.append(4)
    opportunity = features[F_BOMB_OPPORTUNITY]
    if opportunity != BOMB_NONE and (opportunity != BOMB_EMPTY or ALLOW_EMPTY_BOMB):
        allowed.append(5)
    return allowed


def act(self, game_state):
    features = state_to_features(game_state)
    allowed = allowed_actions(features, game_state)

    # full-depth pruning: drop actions after which no survival line exists;
    # if nothing survives, keep the shallow mask so the agent still moves
    if DEEP_PRUNING:
        deep = survivable_actions(game_state)
        pruned = [a for a in allowed if a in deep]
        if pruned:
            allowed = pruned

    if random.random() < self.epsilon:
        return ACTIONS[random.choice(allowed)]

    canon_f, g = canon(features)
    q_values = self.q_table.get(canon_f, np.zeros(len(ACTIONS)))
    best = max(allowed, key=lambda a: q_values[g.apply_action(a)])
    self.logger.debug(f"{features} -> {canon_f} via {g} -> {ACTIONS[best]}")
    return ACTIONS[best]
