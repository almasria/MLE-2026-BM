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
SAFE_BOMB_MULTI_CRATE = "SAFE_BOMB_MULTI_CRATE"
SAFE_BOMB_NEAR_OPPONENT = "SAFE_BOMB_NEAR_OPPONENT"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
FOLLOWED_ESCAPE = "FOLLOWED_ESCAPE"
IGNORED_ESCAPE = "IGNORED_ESCAPE"
SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"
USELESS_BOMB = "USELESS_BOMB"
STILL_IN_DANGER = "STILL_IN_DANGER"
ENTERED_TRAP = "ENTERED_TRAP"
LEFT_TRAP = "LEFT_TRAP"
VULNERABLE = "VULNERABLE"

GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    # CRATE ECONOMY (rebalanced): in the tournament scenario 9 coins hide
    # among ~132 crates, so a crate is worth ~0.07 real points. The old values
    # (5 per crate + 4 + 3 at the drop = +22 for a triple bomb) outweighed a
    # real coin (+10) and taught the agent to walk PAST coins to bomb crates.
    # Value now lives where the points are: revealing a coin, not the crate.
    e.CRATE_DESTROYED: 1.0,
    e.COIN_FOUND: 5.0,
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
    ENTERED_TRAP: -2.0,
    LEFT_TRAP: 1.5,
    VULNERABLE: -1.0,   # v5: unarmed within reach of an armed opponent
    SAFE_BOMB_NEAR_CRATES: 1.5,
    SAFE_BOMB_MULTI_CRATE: 1.5,   # on top: still prefers the richer spot
    SAFE_BOMB_NEAR_OPPONENT: 6.0,
    SUICIDAL_BOMB: -10.0,
    # FOLLOWED_ESCAPE / IGNORED_ESCAPE / SURVIVED_OWN_BOMB were REMOVED.
    # They rewarded escaping danger the agent created itself, which summed to
    # ~+12 per bomb-escape cycle: a farmable reward pump (observed: the agent
    # parked in a corner bombing and escaping forever). Escape execution is
    # now enforced structurally by survivability pruning in the action mask,
    # so no reward pressure is needed. The remaining danger terms are
    # symmetric (+3/-3 = a potential over the danger flag) plus a per-step
    # cost, so ANY danger cycle nets <= 0 unless real value is produced.
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
