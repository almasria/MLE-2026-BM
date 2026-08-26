"""train.py — q_agent v2: Q-learning update + week-3 reward shaping."""

import os
import pickle
import random as _random

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE, canon, USE_SYMMETRY
from .features import state_to_features

# --- Hyperparameters --------------------------------------------------------
ALPHA = float(os.environ.get("Q_AGENT_ALPHA", "0.1"))
GAMMA = float(os.environ.get("Q_AGENT_GAMMA", "0.9"))
EPSILON_START = float(os.environ.get("Q_AGENT_EPS_START", "1.0"))
EPSILON_END = 0.05
EPSILON_DECAY = float(os.environ.get("Q_AGENT_EPS_DECAY", "0.995"))   # W2 lesson: 0.999 leaves eps=0.37 at round 1000 ->
                        # greedy policy under-converged; 0.995 floors by ~600

# --- Custom events (features v2 semantics) ----------------------------------
MOVED_TOWARD_OBJECTIVE = "MOVED_TOWARD_OBJECTIVE"
MOVED_AWAY_FROM_OBJECTIVE = "MOVED_AWAY_FROM_OBJECTIVE"
ESCAPED_DANGER = "ESCAPED_DANGER"        # urgency >0 -> 0
ENTERED_DANGER = "ENTERED_DANGER"        # danger 0 -> 1 (not via own bomb)
SAFE_BOMB_NEAR_CRATES = "SAFE_BOMB_NEAR_CRATES"
SAFE_BOMB_NEAR_OPPONENT = "SAFE_BOMB_NEAR_OPPONENT"
SUICIDAL_BOMB = "SUICIDAL_BOMB"          # bombed with no escape route
FOLLOWED_ESCAPE = "FOLLOWED_ESCAPE"      # in danger, moved along safe_dir
IGNORED_ESCAPE = "IGNORED_ESCAPE"        # in danger, did something else
SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"  # BOMB_EXPLODED without KILLED_SELF
USELESS_BOMB = "USELESS_BOMB"            # safe bomb that hits nothing
STILL_IN_DANGER = "STILL_IN_DANGER"      # every step spent inside a blast zone

GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.CRATE_DESTROYED: 5.0,
    e.COIN_FOUND: 3.0,
    e.KILLED_SELF: -50.0,
    e.GOT_KILLED: -25.0,
    e.SURVIVED_ROUND: 5.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -0.5,
    MOVED_TOWARD_OBJECTIVE: 1.0,
    MOVED_AWAY_FROM_OBJECTIVE: -1.0,
    ESCAPED_DANGER: 3.0,
    ENTERED_DANGER: -3.0,
    STILL_IN_DANGER: -0.75,
    SAFE_BOMB_NEAR_CRATES: 4.0,
    SAFE_BOMB_NEAR_OPPONENT: 6.0,
    SUICIDAL_BOMB: -10.0,
    FOLLOWED_ESCAPE: 2.5,
    IGNORED_ESCAPE: -2.5,
    SURVIVED_OWN_BOMB: 6.0,
    USELESS_BOMB: -2.0,
    e.KILLED_OPPONENT: 25.0,     # stage 4, priced in already
    e.OPPONENT_ELIMINATED: 2.0,
}

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
    with open("training_history.csv", "a") as fh:
        fh.write(f"{self.run_tag},{int(USE_SYMMETRY)},{n},{self.round_reward},"
                 f"{self.round_coins},{self.epsilon:.4f},{len(self.q_table)}\n")
    self.round_reward = 0.0
    self.round_coins = 0

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(self.q_table, f)


def reward_from_events(self, events) -> float:
    return sum(GAME_REWARDS.get(ev, 0.0) for ev in events)
