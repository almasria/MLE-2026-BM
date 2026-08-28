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

from .features import state_to_features
import events as e


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE = "dqn_model.pt"

# features.py returns:
# 0: objective direction
# 1-4: neighbor safety
# 5: urgency
# 6: escape direction
# 7: bomb safety
# 8: crates in range
# 9: opponent direction
# 10: opponent in blast
FEATURE_DIM = 11


class QNetwork(nn.Module):
    """Small MLP: 11 feature values -> 6 action Q-values."""

    def __init__(self, in_dim=FEATURE_DIM, n_actions=len(ACTIONS)):
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
    """Called once before the first round."""

    self.device = torch.device("cpu")

    self.q_net = QNetwork().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.q_net.load_state_dict(
            torch.load(MODEL_FILE, map_location=self.device)
        )
        self.logger.info("Loaded trained DQN weights.")
    else:
        self.logger.info("No saved DQN model found — starting fresh.")

    self.q_net.eval()

    # Test statistics
    self.test_round = None
    self.test_coins = 0
    self.test_start_coins = 0
    self.test_survived = 0

    # Same play-mode exploration idea as q_agent.
    self.epsilon = (
        float(os.environ.get("DQN_PLAY_EPS", "0.02"))
        if not self.train
        else 1.0
    )


def allowed_actions(features, game_state):
    """
    Same safety mask as q_agent.

    Prevent:
      - walking into blocked/unsafe tiles
      - WAIT while urgently trapped in danger
      - BOMB when no bomb is available
      - BOMB when no escape is possible
    """

    # Movement actions
    allowed = [
        i for i in range(4)
        if features[1 + i] == 1
    ]

    # WAIT
    #
    # If urgency == 3 and there is a possible movement action,
    # waiting is considered obviously dangerous.
    if not (features[5] == 3 and allowed):
        allowed.append(4)

    # Cornered -> WAIT remains available
    if not allowed:
        allowed.append(4)

    # BOMB
    bomb_available = game_state['self'][2]

    if bomb_available and features[7] == 1:
        allowed.append(5)

    return allowed


def act(self, game_state: dict) -> str:
    """Choose the highest-valued safe action."""

    # Test coin counter
    if not self.train:
        current_round = game_state['round']
        current_coins = set(game_state['coins'])

        if not hasattr(self, "_test_round"):
            self._test_round = current_round
            self._previous_coins = current_coins
            self._round_coins = 0

        elif current_round != self._test_round:
            self.logger.info(
                f"TEST ROUND {self._test_round}: "
                f"coins_collected={self._round_coins}"
            )

            self._test_round = current_round
            self._previous_coins = current_coins
            self._round_coins = 0

        else:
            collected = len(self._previous_coins - current_coins)

            if collected > 0:
                self._round_coins += collected

            self._previous_coins = current_coins

    # Normal DQN action selection
    features = state_to_features(game_state)

    allowed = allowed_actions(features, game_state)

    if random.random() < self.epsilon:
        return ACTIONS[random.choice(allowed)]

    with torch.no_grad():
        x = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        q_values = self.q_net(x).squeeze(0).cpu().numpy()

    best_action = max(
        allowed,
        key=lambda action: q_values[action]
    )

    self.logger.debug(
        f"features={features} "
        f"allowed={allowed} "
        f"coins={len(game_state['coins'])} "
        f"Q={np.round(q_values, 2)} "
        f"-> {ACTIONS[best_action]}"
    )

    return ACTIONS[best_action]

def end_of_round(self, last_game_state, last_action, events):
    """Print statistics after each test round."""

    if self.train:
        return

    self.test_rounds += 1

    if e.SURVIVED_ROUND in events:
        self.test_wins += 1

    self.logger.info(
        f"TEST ROUND {self.test_rounds}: "
        f"total_coins={self.test_coins} "
        f"wins={self.test_wins}"
    )