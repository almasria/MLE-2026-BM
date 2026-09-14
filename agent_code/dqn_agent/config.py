"""Learning hyperparameters and reward table for RL-Team.

Every hyperparameter can be overridden through an environment variable so
that sweeps (run_sweep.py) need no code changes.

Reward design notes:
  * The tournament scenario hides 9 coins among ~130 crates, so a crate is
    worth about 0.07 real points. Crate rewards are therefore small and the
    value sits on COIN_FOUND (revealing a coin) and COIN_COLLECTED.
  * Danger shaping is symmetric (ENTERED_DANGER = -ESCAPED_DANGER) plus a
    per-step cost, so a self-created danger cycle can never net a profit.
    Escape execution is enforced by the action mask, not by rewards.
  * All custom events depend on the state before/after a step, not on the
    action itself (potential-style shaping).
"""

import os

import events as e

ALPHA = float(os.environ.get("Q_AGENT_ALPHA", "0.1"))
GAMMA = float(os.environ.get("Q_AGENT_GAMMA", "0.9"))
EPSILON_START = float(os.environ.get("Q_AGENT_EPS_START", "1.0"))
EPSILON_END = float(os.environ.get("Q_AGENT_EPS_END", "0.05"))
EPSILON_DECAY = float(os.environ.get("Q_AGENT_EPS_DECAY", "0.995"))

# custom events (raised in train.add_custom_events)
MOVED_TOWARD_OBJECTIVE = "MOVED_TOWARD_OBJECTIVE"
MOVED_AWAY_FROM_OBJECTIVE = "MOVED_AWAY_FROM_OBJECTIVE"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
STILL_IN_DANGER = "STILL_IN_DANGER"
SAFE_BOMB_NEAR_CRATES = "SAFE_BOMB_NEAR_CRATES"
SAFE_BOMB_MULTI_CRATE = "SAFE_BOMB_MULTI_CRATE"
SAFE_BOMB_NEAR_OPPONENT = "SAFE_BOMB_NEAR_OPPONENT"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
ENTERED_TRAP = "ENTERED_TRAP"
LEFT_TRAP = "LEFT_TRAP"
VULNERABLE = "VULNERABLE"

GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.COIN_FOUND: 5.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 25.0,
    e.OPPONENT_ELIMINATED: 2.0,
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
    SAFE_BOMB_NEAR_CRATES: 1.5,
    SAFE_BOMB_MULTI_CRATE: 1.5,
    SAFE_BOMB_NEAR_OPPONENT: 6.0,
    SUICIDAL_BOMB: -10.0,
    USELESS_BOMB: -2.0,
    ENTERED_TRAP: -2.0,
    LEFT_TRAP: 1.5,
    VULNERABLE: -1.0,
}


def q_learning_manifest():
    """Hyperparameters as a dict, for experiment records."""
    return {
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay": EPSILON_DECAY,
    }
