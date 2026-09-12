"""callbacks.py — DQN agent using feature engineering v5."""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from .features import state_to_features
from .featuresv5 import (
    F_URGENCY,
    F_BOMB_OPPORTUNITY,
    BOMB_NONE,
    BOMB_EMPTY,
    URGENCY_IMMINENT,
    survivable_actions,
)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_FILE = os.environ.get(
    "DQN_MODEL_PATH",
    "dqn_model_v5.pt"
)

FEATURE_DIM = 10
ACTION_DIM = len(ACTIONS)

ALLOW_EMPTY_BOMB = (
    os.environ.get("DQN_ALLOW_EMPTY_BOMB", "0") == "1"
)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
        )

    def forward(self, x):
        return self.net(x)


def setup(self):
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    self.model = QNetwork().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.model.load_state_dict(
            torch.load(
                MODEL_FILE,
                map_location=self.device,
                weights_only=True,
            )
        )
        self.logger.info(
            f"Loaded DQN model from {MODEL_FILE}"
        )
    else:
        self.logger.info(
            "No saved DQN model found — starting from scratch."
        )

    self.model.eval()

    if self.train:
        self.epsilon = 1.0
    else:
        self.epsilon = float(
            os.environ.get("DQN_PLAY_EPS", "0.0")
        )

    self.eval_round_coins = 0


def allowed_actions(features, game_state):
    allowed = [
        i
        for i in range(4)
        if features[1 + i] == 1
    ]

    # When danger is imminent, prefer an actual escape move.
    if not (
        features[F_URGENCY] == URGENCY_IMMINENT
        and allowed
    ):
        allowed.append(4)

    if not allowed:
        allowed.append(4)

    opportunity = features[F_BOMB_OPPORTUNITY]

    if (
        opportunity != BOMB_NONE
        and (
            opportunity != BOMB_EMPTY
            or ALLOW_EMPTY_BOMB
        )
    ):
        allowed.append(5)

    return allowed


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    allowed = allowed_actions(
        features,
        game_state,
    )

    # Full-depth death avoidance, same as q_agent.
    deep = survivable_actions(game_state)

    self.logger.debug(
        f"features={features} "
        f"shallow={allowed} "
        f"deep={deep}"
    )

    pruned = [
        a for a in allowed
        if a in deep
    ]

    if pruned:
        allowed = pruned

    # Exploration during training.
    if random.random() < self.epsilon:
        action = random.choice(allowed)

        self.logger.debug(
            f"{features} allowed={allowed} "
            f"epsilon={self.epsilon:.3f} "
            f"-> {ACTIONS[action]}"
        )

        return ACTIONS[action]

    # Neural-network action selection.
    state = torch.tensor(
        features,
        dtype=torch.float32,
        device=self.device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = self.model(state)[0].cpu().numpy()

    best_action = max(
        allowed,
        key=lambda a: q_values[a]
    )

    self.logger.debug(
        f"{features} allowed={allowed} "
        f"Q={np.round(q_values, 2)} "
        f"-> {ACTIONS[best_action]}"
    )

    return ACTIONS[best_action]
