"""Feature set v5: v4 plus an engagement component (10 components).

    9    engagement   0 no opponent within ENGAGE_RADIUS
                      1 advantage: agent armed, nearest opponent has spent its bomb
                      2 no advantage: opponent near but armed, or agent unarmed

Differences from v4 in the objective cascade:
  * a coin within NEAR_COIN steps is taken first
  * during an advantage window the vulnerable opponent is hunted before
    farther coins
  * a contested coin (an opponent is at least as close, an armed opponent
    guards it, or several opponents crowd it) ranks below crate work
  * the "safe but empty" bomb value is folded into "none" (the action mask
    never offers empty bombs, so the distinction carries no information)

Raw state space 5*16*3*5*4*3*3 = 43,200 (2.4x v4); D4 symmetry reduces it
to 7,560 orbits. Reachable states in play stay around one thousand.
"""

import os

from .feature_core import *  # noqa: F401,F403  (layout constants and helpers)
from .feature_core import Perception, bfs_direction_to_nearest, survivable_actions  # noqa: F401

F_ENGAGEMENT = 9
FEATURE_LENGTH = 10
# Q_AGENT_CONTESTED_COINS=0 disables the contested-coin rule (ablation)
CONTESTED_COINS = os.environ.get("Q_AGENT_CONTESTED_COINS", "1") != "0"


def state_to_features(game_state):
    if game_state is None:
        return None
    p = Perception(game_state)
    engagement, vulnerable = p.engagement()

    objective = NO_DIR
    if p.coin_dir != NO_DIR and p.coin_dist <= NEAR_COIN:
        objective = p.coin_dir
    if objective == NO_DIR and engagement == ENGAGE_ADVANTAGE:
        objective = bfs_direction_to_nearest(p.field, (p.x, p.y), vulnerable,
                                             p.walkable_now)
    contested = CONTESTED_COINS and p.coin_dir != NO_DIR and p.coin_contested()
    if objective == NO_DIR and not contested:
        objective = p.coin_dir
    at_spot = False
    if objective == NO_DIR:
        objective, at_spot = p.crate_objective()
    if objective == NO_DIR and not at_spot and contested:
        objective = p.coin_dir                   # a contested coin beats idling
    if objective == NO_DIR and not at_spot:      # on the spot: bomb, don't wander
        objective = p.hunt_objective()

    bomb = p.bomb_opportunity
    if bomb == BOMB_EMPTY and not ALLOW_EMPTY_BOMB:
        bomb = BOMB_NONE

    return (int(objective), *p.neighbours, int(p.urgency), int(p.safe_dir),
            int(bomb), int(p.mobility), int(engagement))
