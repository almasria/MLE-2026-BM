"""Feature version dispatcher.

Q_AGENT_FEATURE_VERSION selects a complete configuration: the feature
function, the matching canonicaliser, the layout constants and the model
file name. callbacks.py and train.py import only from here, so a version
switch is one environment variable and cannot produce a mixed setup.
"""

import os

from .symmetry import LAYOUTS, canonicalize as _canonicalize

FEATURE_VERSION = os.environ.get("Q_AGENT_FEATURE_VERSION", "v5")

if FEATURE_VERSION == "v5":
    from . import featuresv5 as impl
elif FEATURE_VERSION == "v4":
    from . import featuresv4 as impl
else:
    raise ValueError(f"unknown feature version {FEATURE_VERSION!r}")

_layout = LAYOUTS[FEATURE_VERSION]


def canonicalize(features):
    return _canonicalize(features, _layout)


state_to_features = impl.state_to_features
survivable_actions = impl.survivable_actions
FEATURE_LENGTH = impl.FEATURE_LENGTH
MODEL_BASENAME = f"q_table_{FEATURE_VERSION}"

F_OBJECTIVE = impl.F_OBJECTIVE
F_URGENCY = impl.F_URGENCY
F_SAFE_DIR = impl.F_SAFE_DIR
F_BOMB_OPPORTUNITY = impl.F_BOMB_OPPORTUNITY
F_MOBILITY = impl.F_MOBILITY
F_ENGAGEMENT = impl.F_ENGAGEMENT             # None for versions without it
NO_DIR = impl.NO_DIR
URGENCY_SAFE, URGENCY_IMMINENT = impl.URGENCY_SAFE, impl.URGENCY_IMMINENT
BOMB_NONE, BOMB_EMPTY = impl.BOMB_NONE, impl.BOMB_EMPTY
BOMB_CRATES, BOMB_CRATES_MANY = impl.BOMB_CRATES, impl.BOMB_CRATES_MANY
BOMB_OPPONENT = impl.BOMB_OPPONENT
MOBILITY_TRAP = impl.MOBILITY_TRAP
ENGAGE_NO_ADVANTAGE = impl.ENGAGE_NO_ADVANTAGE
