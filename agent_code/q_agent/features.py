"""Stable feature dispatcher for the Q-agent.

Q_AGENT_FEATURE_VERSION selects a COMPLETE configuration: the feature
function, the matching D4 canonicalizer, the layout constants and the model
file name. Everything else imports from here, so switching versions is one
environment variable and can never produce a half-v4/half-v5 agent.
"""

import os

FEATURE_VERSION = os.environ.get("Q_AGENT_FEATURE_VERSION", "v5")

if FEATURE_VERSION == "v5":
    from . import featuresv5 as impl
    from .symmetry import canonicalize_v5 as canonicalize
elif FEATURE_VERSION == "v4":
    from . import featuresv4 as impl
    from .symmetry import canonicalize_v4 as canonicalize
else:
    raise ValueError(
        f"Unknown or incompatible Q-agent feature version: {FEATURE_VERSION!r}"
    )

state_to_features = impl.state_to_features
survivable_actions = impl.survivable_actions
FEATURE_LENGTH = impl.FEATURE_LENGTH

# layout + value constants shared by callbacks.py and train.py
F_OBJECTIVE = impl.F_OBJECTIVE
F_URGENCY = impl.F_URGENCY
F_SAFE_DIR = impl.F_SAFE_DIR
F_BOMB_OPPORTUNITY = impl.F_BOMB_OPPORTUNITY
F_MOBILITY = impl.F_MOBILITY
F_ENGAGEMENT = getattr(impl, "F_ENGAGEMENT", None)          # v5 only
NO_DIR = impl.NO_DIR
URGENCY_SAFE, URGENCY_IMMINENT = impl.URGENCY_SAFE, impl.URGENCY_IMMINENT
BOMB_NONE, BOMB_EMPTY = impl.BOMB_NONE, impl.BOMB_EMPTY
BOMB_CRATES, BOMB_CRATES_MANY = impl.BOMB_CRATES, impl.BOMB_CRATES_MANY
BOMB_OPPONENT = impl.BOMB_OPPONENT
MOBILITY_TRAP = impl.MOBILITY_TRAP
ENGAGE_NO_ADVANTAGE = getattr(impl, "ENGAGE_NO_ADVANTAGE", None)

MODEL_BASENAME = f"q_table_{FEATURE_VERSION}"

__all__ = ["FEATURE_VERSION", "state_to_features", "survivable_actions",
           "canonicalize", "MODEL_BASENAME"]
