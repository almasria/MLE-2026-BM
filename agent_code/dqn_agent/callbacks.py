"""
callbacks.py — Model B: Deep Q-Network agent (stage 1: coin collection).

DESIGN PRINCIPLE: this agent consumes the SAME information as q_agent
(direction to nearest coin + neighbor walkability), just encoded as a
float vector for a neural network instead of a lookup key for a table.
That makes q_agent vs dqn_agent a controlled comparison: same features,
same rewards, different function approximator.

Requires: torch (add to requirements.txt if you submit this agent!).
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from .features import state_to_features as semantic_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STAGE1_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

MODEL_FILE = "dqn_model.pt"

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

# one-hot coin direction (5) + 4 neighbor walkability flags
FEATURE_DIM = 9


class QNetwork(nn.Module):
    """Small MLP: feature vector -> one Q-value per action.

    Deliberately tiny: the tournament runs on CPU with a 0.5 s/step limit,
    and for a 9-dim input a bigger net only slows training down.
    """

    def __init__(self, in_dim=FEATURE_DIM, n_actions=len(STAGE1_ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def setup(self):
    """Called once before the first round. Build/load the network."""
    self.device = torch.device("cpu")
    self.q_net = QNetwork().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.q_net.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
        self.logger.info("Loaded trained DQN weights.")
    else:
        self.logger.info("No saved model found — starting fresh.")
    self.q_net.eval()

    # Small epsilon even in play mode: a deterministic policy in a
    # deterministic world can get stuck in a two-state loop (we verified
    # this failure mode with the tabular agent — same fix here).
    self.epsilon = 0.1 if not self.train else 1.0


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    if random.random() < self.epsilon:
        return random.choice(STAGE1_ACTIONS)

    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        q_values = self.q_net(x).squeeze(0).numpy()
    return STAGE1_ACTIONS[int(np.argmax(q_values))]


# ---------------------------------------------------------------------------
# Features: the SEMANTICS come from the shared features.py (same module as
# q_agent — single source of truth). This wrapper only changes the ENCODING:
# the network needs floats, so the coin direction becomes a one-hot vector.
# ---------------------------------------------------------------------------

def state_to_features(game_state: dict) -> np.ndarray:
    """Shared semantic features -> float32[9] (one-hot dir + 4 walk flags)."""
    if game_state is None:
        return None
    coin_dir, *neighbors = semantic_features(game_state)
    onehot = np.zeros(5, dtype=np.float32)
    onehot[coin_dir] = 1.0
    return np.concatenate([onehot, np.array(neighbors, dtype=np.float32)])
