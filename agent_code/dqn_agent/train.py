"""
train.py — DQN training for Model B.

The two classic tricks that stabilize DQN (both from Mnih et al., 2015):

1. EXPERIENCE REPLAY: transitions go into a buffer; we train on random
   mini-batches from it. Breaks the temporal correlation of consecutive
   samples, which otherwise destabilizes gradient descent, and reuses
   each experience many times.

2. TARGET NETWORK: the TD target r + γ·max Q_target(s') is computed with a
   frozen copy of the network, synced only every N steps. Without it the
   target chases the online network's own updates — a feedback loop that
   commonly diverges.

Reward shaping is IDENTICAL to q_agent — that's what makes the model
comparison in your report controlled.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import events as e
from features import state_to_features
from .callbacks import STAGE1_ACTIONS, MODEL_FILE, QNetwork
# --- Hyperparameters (report material: sweep these!) -----------------------
GAMMA = 0.9
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
MIN_BUFFER = 500          # start learning only once we have this many samples
TARGET_SYNC_EVERY = 1000  # env steps between target-network syncs
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995     # per round (faster than tabular: net generalizes)

# --- Custom events: identical to q_agent ------------------------------------
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

GAME_REWARDS = {
    e.COIN_COLLECTED: 10.0,
    e.INVALID_ACTION: -2.0,
    e.WAITED: -0.5,
    MOVED_TOWARD_COIN: 1.0,
    MOVED_AWAY_FROM_COIN: -1.0,
}


def setup_training(self):
    self.target_net = QNetwork()
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()
    self.q_net.train()

    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
    self.loss_fn = nn.SmoothL1Loss()   # Huber: more robust than MSE to outliers

    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.step_count = 0

    self.epsilon = EPSILON_START
    self.round_reward = 0.0
    self.round_coins = 0
    self.reward_history = []
    self.coins_history = []


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action not in STAGE1_ACTIONS:
        return

    old_f = state_to_features(old_game_state)
    new_f = state_to_features(new_game_state)

    # Custom events — same logic as q_agent (coin one-hot: argmax = direction)
    coin_dir = old_f[0]
    if coin_dir != 4:
        if STAGE1_ACTIONS.index(self_action) == coin_dir:
            events.append(MOVED_TOWARD_COIN)
        elif e.COIN_COLLECTED not in events and e.INVALID_ACTION not in events:
            events.append(MOVED_AWAY_FROM_COIN)

    reward = reward_from_events(self, events)
    self.round_reward += reward
    self.round_coins += events.count(e.COIN_COLLECTED)

    # Store transition and learn from a random mini-batch
    self.buffer.append((old_f, STAGE1_ACTIONS.index(self_action), reward, new_f, False))
    self.step_count += 1
    _learn_step(self)

    if self.step_count % TARGET_SYNC_EVERY == 0:
        self.target_net.load_state_dict(self.q_net.state_dict())


def end_of_round(self, last_game_state, last_action, events):
    reward = reward_from_events(self, events)
    self.round_reward += reward

    if last_action in STAGE1_ACTIONS:
        last_f = state_to_features(last_game_state)
        # terminal transition: done=True -> no bootstrap term in the target
        self.buffer.append((last_f, STAGE1_ACTIONS.index(last_action), reward, last_f, True))
        _learn_step(self)

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    self.reward_history.append(self.round_reward)
    self.coins_history.append(self.round_coins)
    n = len(self.reward_history)
    if n % 100 == 0:
        self.logger.info(
            f"Round {n}: avg reward (last 100) = {np.mean(self.reward_history[-100:]):.1f}, "
            f"avg coins = {np.mean(self.coins_history[-100:]):.2f}, "
            f"epsilon = {self.epsilon:.3f}, buffer = {len(self.buffer)}"
        )
    self.round_reward = 0.0
    self.round_coins = 0

    torch.save(self.q_net.state_dict(), MODEL_FILE)


def _learn_step(self):
    """One gradient step on a random mini-batch (if the buffer is warm)."""
    if len(self.buffer) < MIN_BUFFER:
        return

    batch = random.sample(self.buffer, BATCH_SIZE)
    states = torch.from_numpy(np.stack([t[0] for t in batch])).float()    
    actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
    rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([t[3] for t in batch])).float()
    dones = torch.tensor([t[4] for t in batch], dtype=torch.bool)

    # Q(s,a) for the actions actually taken
    q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD target from the FROZEN network; no bootstrap on terminal states
    with torch.no_grad():
        q_next = self.target_net(next_states).max(dim=1).values
        q_next[dones] = 0.0
        q_target = rewards + GAMMA * q_next

    loss = self.loss_fn(q_pred, q_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


def reward_from_events(self, events) -> float:
    return sum(GAME_REWARDS.get(ev, 0.0) for ev in events)
