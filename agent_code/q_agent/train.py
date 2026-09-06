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
    GAME_REWARDS,
    GAMMA,
    MOVED_AWAY_FROM_OBJECTIVE,
    MOVED_TOWARD_OBJECTIVE,
    SAFE_BOMB_MULTI_CRATE,
    SAFE_BOMB_NEAR_CRATES,
    SAFE_BOMB_NEAR_OPPONENT,
    ENTERED_TRAP,
    LEFT_TRAP,
    VULNERABLE,
    STILL_IN_DANGER,
    SUICIDAL_BOMB,
    USELESS_BOMB,
)
from .features import state_to_features

TRAINING_HISTORY_FILE = os.environ.get(
    "Q_AGENT_HISTORY_PATH", "training_history.csv"
)

# ground-truth movement from the framework's own events (a failed move
# raises INVALID_ACTION instead, so this never miscounts)
MOVED_EVENTS = {e.MOVED_UP: 0, e.MOVED_RIGHT: 1, e.MOVED_DOWN: 2, e.MOVED_LEFT: 3}

# v4 feature layout is imported symbolically, so index drift cannot happen
from .features import (BOMB_CRATES, BOMB_CRATES_MANY, BOMB_EMPTY, BOMB_NONE,
                       BOMB_OPPONENT, ENGAGE_NO_ADVANTAGE, F_ENGAGEMENT,
                       F_BOMB_OPPORTUNITY, F_MOBILITY, F_OBJECTIVE,
                       F_SAFE_DIR, F_URGENCY, MOBILITY_TRAP, NO_DIR,
                       URGENCY_SAFE)

F_OBJ, F_DANGER, F_SAFE = F_OBJECTIVE, F_URGENCY, F_SAFE_DIR


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

    if old_f[F_DANGER] > 0 and new_f is not None and new_f[F_DANGER] == 0:
        events.append(ESCAPED_DANGER)
    # NOTE: no exemption for action == 'BOMB'. Entering danger costs -3 even
    # when self-inflicted; symmetric with ESCAPED_DANGER +3, the pair cancels
    # over any cycle (potential-based shaping), leaving only the per-step
    # STILL_IN_DANGER cost. Self-created danger is thus never profitable
    # by itself -- only its PRODUCTS (crates, opponents) pay.
    if old_f[F_DANGER] == 0 and new_f is not None and new_f[F_DANGER] > 0:
        events.append(ENTERED_DANGER)

    if e.BOMB_DROPPED in events:
        opportunity = old_f[F_BOMB_OPPORTUNITY]
        if opportunity == BOMB_NONE:
            events.append(SUICIDAL_BOMB)
        elif opportunity == BOMB_OPPONENT:
            events.append(SAFE_BOMB_NEAR_OPPONENT)
        elif opportunity in (BOMB_CRATES, BOMB_CRATES_MANY):
            # fires while the agent is merely PASSING a crate on its way to a
            # coin, which is what makes opportunistic bombing learnable
            events.append(SAFE_BOMB_NEAR_CRATES)
            if opportunity == BOMB_CRATES_MANY:
                events.append(SAFE_BOMB_MULTI_CRATE)
        else:
            events.append(USELESS_BOMB)

    # v5: standing UNARMED next to an ARMED opponent is a bad state to be in
    # (state-only penalty, potential-style: no cycle can farm it)
    if F_ENGAGEMENT is not None and new_f is not None \
            and new_f[F_ENGAGEMENT] == ENGAGE_NO_ADVANTAGE \
            and new_f[F_BOMB_OPPORTUNITY] == BOMB_NONE:
        events.append(VULNERABLE)

    # trap pressure: mobility collapses when walls, crates, bombs or opponents
    # close in, and leaving early is much cheaper than escaping later
    if new_f is not None:
        if old_f[F_MOBILITY] != MOBILITY_TRAP and new_f[F_MOBILITY] == MOBILITY_TRAP:
            events.append(ENTERED_TRAP)
        elif old_f[F_MOBILITY] == MOBILITY_TRAP and new_f[F_MOBILITY] != MOBILITY_TRAP:
            events.append(LEFT_TRAP)

    # objective shaping only while safe — while in danger, escaping rules
    if old_f[F_DANGER] > 0 and new_f is not None and new_f[F_DANGER] > 0:
        events.append(STILL_IN_DANGER)

    if old_f[F_DANGER] == URGENCY_SAFE and old_f[F_OBJ] != NO_DIR and moved_dir is not None:
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
    # Optional diagnostic (Q_AGENT_DEATH_LOG=1): record the circumstances of
    # every self-kill. This is how the opponent-threat model was discovered:
    # 24/24 deaths had an armed opponent within 2 tiles and no escape left.
    if os.environ.get("Q_AGENT_DEATH_LOG") == "1" and e.KILLED_SELF in events \
            and last_game_state is not None:
        gs = last_game_state
        x, y = gs['self'][3]
        opp = [o[3] for o in gs['others']]
        dmin = min((abs(ox - x) + abs(oy - y) for ox, oy in opp), default=99)
        other_bombs = sum(1 for pos, _ in gs['bombs'] if pos != (x, y))
        lf = state_to_features(gs)
        with open("death_log.csv", "a") as fh:
            fh.write(f"{gs['step']},{dmin},{other_bombs},{lf[F_DANGER]},"
                     f"{lf[F_SAFE]},{lf[F_MOBILITY]},{last_action}\n")
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
