"""Feature set v4: 9 components.

    0    objective_dir      nearest coin, else best crate bomb spot, else
                            nearest opponent (endgame); 4 = none
    1-4  neighbours         1 if that adjacent tile can be entered safely
    5    urgency            0 safe / 1 lethal in 2+ steps / 2 lethal within 1
    6    safe_dir           first step of a time-aware escape; 4 = none needed
    7    bomb_opportunity   0 none / 1 empty / 2 crates / 3 many crates / 4 opponent
    8    mobility           0 trapped / 1 tight / 2 open

Raw state space 5*16*3*5*5*3 = 18,000; D4 symmetry reduces it further.
Retained as the ablation baseline for v5.
"""

from .feature_core import *  # noqa: F401,F403  (layout constants and helpers)
from .feature_core import Perception, bfs_direction_to_nearest, survivable_actions  # noqa: F401

F_ENGAGEMENT = None
ENGAGE_NO_ADVANTAGE = None
FEATURE_LENGTH = 9


def state_to_features(game_state):
    if game_state is None:
        return None
    p = Perception(game_state)

    objective = p.coin_dir
    at_spot = False
    if objective == NO_DIR:
        objective, at_spot = p.crate_objective()
    if objective == NO_DIR and not at_spot:      # on the spot: bomb, don't wander
        objective = p.hunt_objective()

    return (int(objective), *p.neighbours, int(p.urgency), int(p.safe_dir),
            int(p.bomb_opportunity), int(p.mobility))
