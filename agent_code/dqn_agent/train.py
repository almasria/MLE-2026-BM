"""
train.py — DQN training for Model B.

DQN with:
- experience replay
- target network
- the same W3 reward shaping as q_agent
- the same 11 features as q_agent
- the same 6 actions as q_agent
- the same safety action masking as q_agent
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import events as e

from .features import state_to_features
from .callbacks import (
    ACTIONS,
    MODEL_FILE,
    QNetwork,
    allowed_actions,
)


# Hyperparameters

GAMMA = 0.9
LEARNING_RATE = 5e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
MIN_BUFFER = 500
TARGET_SYNC_EVERY = 2000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995


# W3 reward shaping

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
    e.COIN_COLLECTED: 15.0,
    e.CRATE_DESTROYED: 3.0,
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
    USELESS_BOMB: -4.0,

    e.KILLED_OPPONENT: 25.0,
    e.OPPONENT_ELIMINATED: 2.0,
}


# Framework movement events

MOVED_EVENTS = {
    e.MOVED_UP: 0,
    e.MOVED_RIGHT: 1,
    e.MOVED_DOWN: 2,
    e.MOVED_LEFT: 3,
}


# Feature indices

F_OBJ = 0
F_DANGER = 5
F_SAFE = 6
F_BOMBSAFE = 7
F_CRATES = 8
F_OPP = 9
F_OPPBLAST = 10


# Custom reward events
def add_custom_events(old_f, action, new_f, events):
    """Same reward shaping logic as q_agent."""

    moved_dir = next(
        (
            direction
            for event, direction in MOVED_EVENTS.items()
            if event in events
        ),
        None
    )

    # Escape discipline
    if old_f[F_DANGER] > 0 and old_f[F_SAFE] != 4:
        if moved_dir == old_f[F_SAFE]:
            events.append(FOLLOWED_ESCAPE)
        else:
            events.append(IGNORED_ESCAPE)

    # Bomb survival
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(SURVIVED_OWN_BOMB)


    # Danger transitions
    if (
        old_f[F_DANGER] > 0
        and new_f is not None
        and new_f[F_DANGER] == 0
    ):
        events.append(ESCAPED_DANGER)

    if (
        old_f[F_DANGER] == 0
        and new_f is not None
        and new_f[F_DANGER] > 0
        and action != "BOMB"
    ):
        events.append(ENTERED_DANGER)

    # Bomb quality
    if e.BOMB_DROPPED in events:

        if old_f[F_BOMBSAFE] == 0:
            events.append(SUICIDAL_BOMB)

        else:
            if old_f[F_CRATES] > 0:
                events.append(SAFE_BOMB_NEAR_CRATES)

            if old_f[F_OPPBLAST] == 1:
                events.append(SAFE_BOMB_NEAR_OPPONENT)

            if (
                old_f[F_CRATES] == 0
                and old_f[F_OPPBLAST] == 0
            ):
                events.append(USELESS_BOMB)

    # Still in danger
    if (
        old_f[F_DANGER] > 0
        and new_f is not None
        and new_f[F_DANGER] > 0
    ):
        events.append(STILL_IN_DANGER)

    # Objective shaping while safe
    if (
        old_f[F_DANGER] == 0
        and old_f[F_OBJ] != 4
        and moved_dir is not None
    ):

        if moved_dir == old_f[F_OBJ]:
            events.append(MOVED_TOWARD_OBJECTIVE)

        elif e.COIN_COLLECTED not in events:
            events.append(MOVED_AWAY_FROM_OBJECTIVE)


# Training setup
def setup_training(self):

    self.target_net = QNetwork()

    self.target_net.load_state_dict(
        self.q_net.state_dict()
    )

    self.target_net.eval()
    self.q_net.train()

    self.optimizer = torch.optim.Adam(
        self.q_net.parameters(),
        lr=LEARNING_RATE
    )

    self.loss_fn = nn.SmoothL1Loss()

    self.buffer = deque(
        maxlen=BUFFER_SIZE
    )

    self.step_count = 0

    self.epsilon = EPSILON_START

    self.round_reward = 0.0
    self.round_coins = 0

    self.reward_history = []
    self.coins_history = []


# Game event
def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events
):

    if (
        old_game_state is None
        or self_action not in ACTIONS
    ):
        return

    old_f = state_to_features(old_game_state)
    new_f = state_to_features(new_game_state)

    add_custom_events(
        old_f,
        self_action,
        new_f,
        events
    )

    reward = reward_from_events(
        self,
        events
    )

    self.round_reward += reward

    self.round_coins += events.count(
        e.COIN_COLLECTED
    )

    next_allowed = allowed_actions(
        new_f,
        new_game_state
    )

    # Store transition
    self.buffer.append(
        (
            old_f,
            ACTIONS.index(self_action),
            reward,
            new_f,
            next_allowed,
            False,
        )
    )

    self.step_count += 1

    _learn_step(self)

    # Periodically synchronize target network
    if self.step_count % TARGET_SYNC_EVERY == 0:

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )


# End of round
def end_of_round(
    self,
    last_game_state,
    last_action,
    events
):

    reward = reward_from_events(
        self,
        events
    )

    self.round_reward += reward

    if last_action in ACTIONS:

        last_f = state_to_features(
            last_game_state
        )

        # Terminal state has no future actions.
        self.buffer.append(
            (
                last_f,
                ACTIONS.index(last_action),
                reward,
                last_f,
                [],
                True,
            )
        )

        _learn_step(self)

    # Epsilon decay
    self.epsilon = max(
        EPSILON_END,
        self.epsilon * EPSILON_DECAY
    )

    self.reward_history.append(
        self.round_reward
    )

    self.coins_history.append(
        self.round_coins
    )

    n = len(
        self.reward_history
    )

    if n % 10 == 0:

        self.logger.info(
            f"Round {n}: "
            f"avg reward (last 100) = "
            f"{np.mean(self.reward_history[-100:]):.1f}, "
            f"avg coins = "
            f"{np.mean(self.coins_history[-100:]):.2f}, "
            f"epsilon = "
            f"{self.epsilon:.3f}, "
            f"buffer = "
            f"{len(self.buffer)}"
        )

    self.round_reward = 0.0
    self.round_coins = 0

    # Save network
    torch.save(
        self.q_net.state_dict(),
        MODEL_FILE
    )


# DQN learning step
def _learn_step(self):

    if len(self.buffer) < MIN_BUFFER:
        return

    batch = random.sample(
        self.buffer,
        BATCH_SIZE
    )

    states = torch.from_numpy(
        np.stack(
            [t[0] for t in batch]
        )
    ).float()

    actions = torch.tensor(
        [t[1] for t in batch],
        dtype=torch.long
    )

    rewards = torch.tensor(
        [t[2] for t in batch],
        dtype=torch.float32
    )

    next_states = torch.from_numpy(
        np.stack(
            [t[3] for t in batch]
        )
    ).float()

    next_allowed = [
        t[4]
        for t in batch
    ]

    dones = torch.tensor(
        [t[5] for t in batch],
        dtype=torch.bool
    )

    # Current Q(s,a)
    q_pred = (
        self.q_net(states)
        .gather(
            1,
            actions.unsqueeze(1)
        )
        .squeeze(1)
    )

    with torch.no_grad():

        q_next_online = self.q_net(next_states)

        q_next_target = self.target_net(next_states)

        q_next = torch.zeros(
            BATCH_SIZE,
            dtype=torch.float32
        )

        for i in range(BATCH_SIZE):

            if dones[i]:
                q_next[i] = 0.0
                continue

            allowed = next_allowed[i]

            if not allowed:
                q_next[i] = 0.0
                continue

            best_action = allowed[
                torch.argmax(q_next_online[i, allowed]).item()
            ]

            q_next[i] = q_next_target[i, best_action]

        q_target = rewards + GAMMA * q_next

    # Gradient update
    loss = self.loss_fn(
        q_pred,
        q_target
    )

    self.optimizer.zero_grad()

    loss.backward()

    self.optimizer.step()


# Reward calculation
def reward_from_events(self, events) -> float:

    return sum(
        GAME_REWARDS.get(
            event,
            0.0
        )
        for event in events
    )