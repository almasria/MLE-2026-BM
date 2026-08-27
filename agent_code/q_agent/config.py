"""Single source of truth for Q-agent learning and reward configuration."""

import os

import events as e


ALPHA = float(os.environ.get("Q_AGENT_ALPHA", "0.1"))
GAMMA = float(os.environ.get("Q_AGENT_GAMMA", "0.9"))
EPSILON_START = float(os.environ.get("Q_AGENT_EPS_START", "1.0"))
EPSILON_END = float(os.environ.get("Q_AGENT_EPS_END", "0.05"))
EPSILON_DECAY = float(os.environ.get("Q_AGENT_EPS_DECAY", "0.995"))

MOVED_TOWARD_OBJECTIVE = "MOVED_TOWARD_OBJECTIVE"
MOVED_AWAY_FROM_OBJECTIVE = "MOVED_AWAY_FROM_OBJECTIVE"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
SAFE_BOMB_NEAR_CRATES = "SAFE_BOMB_NEAR_CRATES"
SAFE_BOMB_NEAR_OPPONENT = "SAFE_BOMB_NEAR_OPPONENT"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
FOLLOWED_ESCAPE = "FOLLOWED_ESCAPE"
IGNORED_ESCAPE = "IGNORED_ESCAPE"
SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"
USELESS_BOMB = "USELESS_BOMB"
STILL_IN_DANGER = "STILL_IN_DANGER"

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
    e.KILLED_OPPONENT: 25.0,
    e.OPPONENT_ELIMINATED: 2.0,
}


def q_learning_manifest():
    """Return JSON-serializable learning parameters for experiment records."""
    return {
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay": EPSILON_DECAY,
    }
