"""Stable feature dispatcher for the Q-agent.

Experiments select a committed feature implementation with
``Q_AGENT_FEATURE_VERSION``.  Keeping this module stable prevents the
experiment runner from copying files over tracked source code.
"""

import os


FEATURE_VERSION = os.environ.get("Q_AGENT_FEATURE_VERSION", "v4")

if FEATURE_VERSION == "v4":
    from .featuresv4 import state_to_features
elif FEATURE_VERSION == "v3":
    from .featuresv3 import state_to_features
else:
    raise ValueError(
        f"Unknown or incompatible Q-agent feature version: {FEATURE_VERSION!r}"
    )


__all__ = ["FEATURE_VERSION", "state_to_features"]
