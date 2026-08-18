"""
train.py — Q-learning training logic for the stage-1 coin collector.

Framework contract (called by the environment when run with --train 1):
    setup_training(self)
    game_events_occurred(self, old_game_state, self_action, new_game_state, events)
    end_of_round(self, last_game_state, last_action, events)

The learning rule is textbook tabular Q-learning:
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
"""

import pickle
from collections import defaultdict

import numpy as np

import events as e
from .callbacks import STAGE1_ACTIONS, MODEL_FILE
from .features import state_to_features
from .symmetry import canonicalize_v1

# --- Hyperparameters -------------------------------------------------------
ALPHA = 0.1            # learning rate
GAMMA = 0.9            # discount factor
EPSILON_START = 1.0    # initial exploration
EPSILON_END = 0.05     # floor
EPSILON_DECAY = 0.999  # multiplied every round (~2000 rounds to reach floor)

# --- Custom events (our reward-shaping lever) -------------------------------
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -0.5,
    MOVED_TOWARD_COIN: 1.0,
    MOVED_AWAY_FROM_COIN: -1.0,
    # Stage 2+ additions (kept here as a reminder of where they go):
    # e.CRATE_DESTROYED: 3.0, e.COIN_FOUND: 2.0,
    # e.KILLED_OPPONENT: 25.0, e.KILLED_SELF: -50.0,
    # e.GOT_KILLED: -25.0, e.SURVIVED_ROUND: 5.0,
}


def setup_training(self):
    self.epsilon = EPSILON_START
    self.round_reward = 0.0
    self.reward_history = []          # per-round totals -> plot this!
    self.coins_history = []
    self.round_coins = 0


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called once per step during training. This is where learning happens."""
    if old_game_state is None:
        return

    # ---- 1. Add custom events (reward shaping) ----------------------------
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # old_features[0] is the BFS direction to the nearest coin (0..3, 4 = none).
    # If the chosen action matches it, the agent moved toward the coin.
    coin_dir = old_features[0]
    if coin_dir != 4 and self_action in STAGE1_ACTIONS:
        if STAGE1_ACTIONS.index(self_action) == coin_dir:
            events.append(MOVED_TOWARD_COIN)
        elif e.COIN_COLLECTED not in events and e.INVALID_ACTION not in events:
            events.append(MOVED_AWAY_FROM_COIN)

    # ---- 2. Compute reward -------------------------------------------------
    reward = reward_from_events(self, events)
    self.round_reward += reward
    self.round_coins += events.count(e.COIN_COLLECTED)

    # ---- 3. Q-learning update ---------------------------------------------
    if self_action not in STAGE1_ACTIONS:
        return  # ignore actions outside our restricted set (shouldn't happen)

    # D4 canonical update: both states go to their orbit representatives;
    # the REAL action taken is mapped forward into the old state's canonical
    # frame. (The custom-event logic above deliberately keeps using the RAW
    # features — "moved toward the coin" is a real-world statement.)
    canon_old, g_old = canonicalize_v1(old_features)
    canon_new, _ = canonicalize_v1(new_features)
    a = g_old.apply_action(STAGE1_ACTIONS.index(self_action))
    q_old = self.q_table.setdefault(canon_old, np.zeros(len(STAGE1_ACTIONS)))
    q_next = self.q_table.get(canon_new, np.zeros(len(STAGE1_ACTIONS)))

    td_target = reward + GAMMA * np.max(q_next)
    q_old[a] += ALPHA * (td_target - q_old[a])


def end_of_round(self, last_game_state, last_action, events):
    """Called at the end of each round: final update, decay epsilon, save."""
    # Final transition (no successor state -> target is just the reward)
    reward = reward_from_events(self, events)
    self.round_reward += reward

    last_features = state_to_features(last_game_state)
    if last_action in STAGE1_ACTIONS:
        canon_last, g_last = canonicalize_v1(last_features)
        a = g_last.apply_action(STAGE1_ACTIONS.index(last_action))
        q = self.q_table.setdefault(canon_last, np.zeros(len(STAGE1_ACTIONS)))
        q[a] += ALPHA * (reward - q[a])

    # Decay exploration
    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # Book-keeping
    self.reward_history.append(self.round_reward)
    self.coins_history.append(self.round_coins)
    n = len(self.reward_history)
    if n % 100 == 0:
        avg_r = np.mean(self.reward_history[-100:])
        avg_c = np.mean(self.coins_history[-100:])
        self.logger.info(
            f"Round {n}: avg reward (last 100) = {avg_r:.1f}, "
            f"avg coins = {avg_c:.2f}, epsilon = {self.epsilon:.3f}, "
            f"Q-table size = {len(self.q_table)}"
        )
    self.round_reward = 0.0
    self.round_coins = 0

    # Save the model every round (cheap for a small Q-table)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(self.q_table, f)


def reward_from_events(self, events) -> float:
    """Map the list of event strings to a scalar reward."""
    reward = sum(GAME_REWARDS.get(ev, 0.0) for ev in events)
    self.logger.debug(f"Events {events} -> reward {reward}")
    return reward
