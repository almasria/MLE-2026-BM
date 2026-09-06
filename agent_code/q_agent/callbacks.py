"""callbacks.py — q_agent v2: tabular Q-learning, full action set, D4 canon."""

import os
import pickle
import random

import numpy as np

from .features import (BOMB_EMPTY, BOMB_NONE, F_BOMB_OPPORTUNITY, F_MOBILITY,
                       F_URGENCY, MOBILITY_TRAP, MODEL_BASENAME, URGENCY_IMMINENT,
                       canonicalize, state_to_features, survivable_actions)
from .symmetry import IDENTITY

# Empty bombs (survivable, hit nothing) are masked by default: they were the
# near-wall drops observed vs opponents.  Bombs covering an opponent rank
# BOMB_OPPONENT, so anti-opponent play is unaffected.  The cost is the rare
# pre-placement tactic (bombing where an opponent WILL be); re-enable with
# Q_AGENT_ALLOW_EMPTY_BOMB=1 to measure that trade-off.
ALLOW_EMPTY_BOMB = os.environ.get("Q_AGENT_ALLOW_EMPTY_BOMB", "0") == "1"

# --- Ablation switch (see week-2 experiment) --------------------------------
USE_SYMMETRY = os.environ.get("Q_AGENT_SYMMETRY", "1") != "0"


def canon(features):
    return canonicalize(features) if USE_SYMMETRY else (features, IDENTITY)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DEFAULT_MODEL_FILE = f"{MODEL_BASENAME}_{'sym' if USE_SYMMETRY else 'plain'}.pkl"
MODEL_FILE = os.environ.get("Q_AGENT_MODEL_PATH", DEFAULT_MODEL_FILE)


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
    # WAIT is legal — except standing still with a blast about to arrive and
    # somewhere to run, which is certain death
    if not (features[F_URGENCY] == URGENCY_IMMINENT and allowed):
        allowed.append(4)
    if not allowed:                                     # cornered: allow WAIT
        allowed.append(4)
    # BOMB only when it is available AND survivable; bomb_opportunity already
    # folds both conditions in (0 means unavailable or unsurvivable).
    opportunity = features[F_BOMB_OPPORTUNITY]
    if opportunity != BOMB_NONE and (opportunity != BOMB_EMPTY or ALLOW_EMPTY_BOMB):
        allowed.append(5)
    # NOTE: mobility deliberately does NOT gate bombing here.  Measured: gating
    # it cost 15 crates/round, because the tightest spots are exactly the
    # crate-dense ones worth bombing, and bomb_opportunity has already proved
    # an escape exists.  Mobility stays a feature (and a shaped event) so the
    # agent can learn when a tight spot is worth it.
    return allowed


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    allowed = allowed_actions(features, game_state)
    # full-depth certain-death pruning: drop actions with NO survival line.
    # If nothing survives (truly doomed), keep the shallow mask as-is.
    deep = survivable_actions(game_state)
    pruned = [a for a in allowed if a in deep]
    if pruned:
        allowed = pruned

    if random.random() < self.epsilon:
        return ACTIONS[random.choice(allowed)]

    canon_f, g = canon(features)
    q_values = self.q_table.get(canon_f, np.zeros(len(ACTIONS)))
    best_real = max(allowed, key=lambda a: q_values[g.apply_action(a)])
    self.logger.debug(f"{features} -> {canon_f} via {g} -> {ACTIONS[best_real]}")
    return ACTIONS[best_real]
