"""train.py — q_agent v2: Q-learning update + week-3 reward shaping."""

import os
import pickle
import random as _random

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE, canon, USE_SYMMETRY
from .config import (
    ALPHA,
    ENTERED_DANGER,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    ESCAPED_DANGER,
    FOLLOWED_ESCAPE,
    GAME_REWARDS,
    GAMMA,
    IGNORED_ESCAPE,
    MOVED_AWAY_FROM_OBJECTIVE,
    MOVED_TOWARD_OBJECTIVE,
    SAFE_BOMB_NEAR_CRATES,
    SAFE_BOMB_NEAR_OPPONENT,
    STILL_IN_DANGER,
    SUICIDAL_BOMB,
    SURVIVED_OWN_BOMB,
    USELESS_BOMB,
)
from .features import state_to_features

TRAINING_HISTORY_FILE = os.environ.get(
    "Q_AGENT_HISTORY_PATH", "training_history.csv"
)

# ground-truth movement from the framework's own events (a failed move
# raises INVALID_ACTION instead, so this never miscounts)
MOVED_EVENTS = {e.MOVED_UP: 0, e.MOVED_RIGHT: 1, e.MOVED_DOWN: 2, e.MOVED_LEFT: 3}

# v2 feature indices (keep in sync with features.py!)
F_OBJ, F_DANGER, F_SAFE, F_BOMBSAFE, F_CRATES, F_OPP, F_OPPBLAST = 0, 5, 6, 7, 8, 9, 10


def setup_training(self):
    seed = os.environ.get("Q_AGENT_SEED")
    if seed is not None:
        _random.seed(int(seed)); np.random.seed(int(seed))
    self.run_tag = os.environ.get("Q_AGENT_RUN_TAG", "default")
    self.epsilon = EPSILON_START
    self.round_reward = 0.0
    self.round_coins = 0
    self.reward_history, self.coins_history = [], []


def add_custom_events(old_f, action, new_f, events):
    """Derive shaping events from the v2 features. RAW frame on purpose:
    'toward the objective' is a real-world statement (see symmetry notes)."""
    moved_dir = next((d for ev, d in MOVED_EVENTS.items() if ev in events), None)

    # escape discipline: while in danger with a known way out, following it
    # is rewarded and anything else is punished -- nothing competes with escape
    if old_f[F_DANGER] > 0 and old_f[F_SAFE] != 4:
        if moved_dir == old_f[F_SAFE]:
            events.append(FOLLOWED_ESCAPE)
        else:
            events.append(IGNORED_ESCAPE)

    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(SURVIVED_OWN_BOMB)
    if old_f[F_DANGER] > 0 and new_f is not None and new_f[F_DANGER] == 0:
        events.append(ESCAPED_DANGER)
    if old_f[F_DANGER] == 0 and new_f is not None and new_f[F_DANGER] > 0 \
            and action != 'BOMB':
        events.append(ENTERED_DANGER)

    if e.BOMB_DROPPED in events:
        if old_f[F_BOMBSAFE] == 0:
            events.append(SUICIDAL_BOMB)
        else:
            if old_f[F_CRATES] > 0:
                events.append(SAFE_BOMB_NEAR_CRATES)
            if old_f[F_OPPBLAST] == 1:
                events.append(SAFE_BOMB_NEAR_OPPONENT)
            if old_f[F_CRATES] == 0 and old_f[F_OPPBLAST] == 0:
                events.append(USELESS_BOMB)

    # objective shaping only while safe — while in danger, escaping rules
    if old_f[F_DANGER] > 0 and new_f is not None and new_f[F_DANGER] > 0:
        events.append(STILL_IN_DANGER)

    if old_f[F_DANGER] == 0 and old_f[F_OBJ] != 4 and moved_dir is not None:
        if moved_dir == old_f[F_OBJ]:
            events.append(MOVED_TOWARD_OBJECTIVE)
        elif e.COIN_COLLECTED not in events:
            events.append(MOVED_AWAY_FROM_OBJECTIVE)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in ACTIONS:
        return
    old_f = state_to_features(old_game_state)
    new_f = state_to_features(new_game_state)
    add_custom_events(old_f, self_action, new_f, events)

    reward = reward_from_events(self, events)
    self.round_reward += reward
    self.round_coins += events.count(e.COIN_COLLECTED)

    canon_old, g_old = canon(old_f)
    canon_new, _ = canon(new_f)
    a = g_old.apply_action(ACTIONS.index(self_action))
    q_old = self.q_table.setdefault(canon_old, np.zeros(len(ACTIONS)))
    q_next = self.q_table.get(canon_new, np.zeros(len(ACTIONS)))
    q_old[a] += ALPHA * (reward + GAMMA * np.max(q_next) - q_old[a])


def end_of_round(self, last_game_state, last_action, events):
    reward = reward_from_events(self, events)
    self.round_reward += reward

    if last_action in ACTIONS:
        last_f = state_to_features(last_game_state)
        canon_last, g_last = canon(last_f)
        a = g_last.apply_action(ACTIONS.index(last_action))
        q = self.q_table.setdefault(canon_last, np.zeros(len(ACTIONS)))
        q[a] += ALPHA * (reward - q[a])

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    self.reward_history.append(self.round_reward)
    self.coins_history.append(self.round_coins)
    n = len(self.reward_history)
    if n % 100 == 0:
        self.logger.info(
            f"Round {n}: avg reward (last 100) = {np.mean(self.reward_history[-100:]):.1f}, "
            f"avg coins = {np.mean(self.coins_history[-100:]):.2f}, "
            f"epsilon = {self.epsilon:.3f}, Q-table = {len(self.q_table)}")
    with open(TRAINING_HISTORY_FILE, "a") as fh:
        fh.write(f"{self.run_tag},{int(USE_SYMMETRY)},{n},{self.round_reward},"
                 f"{self.round_coins},{self.epsilon:.4f},{len(self.q_table)}\n")
    self.round_reward = 0.0
    self.round_coins = 0

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(self.q_table, f)


def reward_from_events(self, events) -> float:
    return sum(GAME_REWARDS.get(ev, 0.0) for ev in events)
