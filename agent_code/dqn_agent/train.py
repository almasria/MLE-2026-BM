"""DQN training with feature engineering v4."""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e

from .callbacks import (
    ACTIONS,
    QNetwork,
    allowed_actions,
)
from .features import state_to_features

from .config import (
    ENTERED_DANGER,
    ESCAPED_DANGER,
    GAME_REWARDS,
    MOVED_AWAY_FROM_OBJECTIVE,
    MOVED_TOWARD_OBJECTIVE,
    SAFE_BOMB_MULTI_CRATE,
    SAFE_BOMB_NEAR_CRATES,
    SAFE_BOMB_NEAR_OPPONENT,
    ENTERED_TRAP,
    LEFT_TRAP,
    STILL_IN_DANGER,
    SUICIDAL_BOMB,
    USELESS_BOMB,
)

from .featuresv5 import (
    BOMB_CRATES,
    BOMB_CRATES_MANY,
    BOMB_NONE,
    BOMB_OPPONENT,
    F_BOMB_OPPORTUNITY,
    F_MOBILITY,
    F_OBJECTIVE,
    F_URGENCY,
    NO_DIR,
    URGENCY_SAFE,
    MOBILITY_TRAP,
    survivable_actions,
)


# Hyperparameters
GAMMA = float(
    os.environ.get("DQN_GAMMA", "0.90")
)

LR = float(
    os.environ.get("DQN_LR", "1e-3")
)

BATCH_SIZE = int(
    os.environ.get("DQN_BATCH_SIZE", "64")
)

BUFFER_SIZE = int(
    os.environ.get("DQN_BUFFER_SIZE", "50000")
)

MIN_BUFFER = int(
    os.environ.get("DQN_MIN_BUFFER", "500")
)

TARGET_UPDATE = int(
    os.environ.get("DQN_TARGET_UPDATE", "1000")
)

EPSILON_START = float(
    os.environ.get("DQN_EPSILON_START", "1.0")
)

EPSILON_END = float(
    os.environ.get("DQN_EPSILON_END", "0.05")
)

EPSILON_DECAY = float(
    os.environ.get("DQN_EPSILON_DECAY", "0.998")
)


# Movement events
MOVED_EVENTS = {
    e.MOVED_UP: 0,
    e.MOVED_RIGHT: 1,
    e.MOVED_DOWN: 2,
    e.MOVED_LEFT: 3,
}


# Replay-buffer helper
def get_allowed_next_actions(game_state):
    """
    Get the actions that the v4 safety system considers survivable.

    First apply the shallow feature-based action mask, then apply the
    full-depth survivable_actions() pruning.

    Returns action indices:
        0 = UP
        1 = RIGHT
        2 = DOWN
        3 = LEFT
        4 = WAIT
        5 = BOMB
    """

    if game_state is None:
        return []

    features = state_to_features(game_state)

    allowed = allowed_actions(
        features,
        game_state,
    )

    deep = survivable_actions(
        game_state
    )

    pruned = [
        action
        for action in allowed
        if action in deep
    ]

    # Same fallback behavior as q_agent:
    # only use the deep mask if it leaves something available.
    if pruned:
        allowed = pruned

    return allowed


# Training setup
def setup_training(self):

    seed = os.environ.get("DQN_SEED")

    if seed is not None:
        seed = int(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    self.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    self.model = QNetwork().to(self.device)

    model_path = "dqn_model_v5.pt"

    if os.path.isfile(model_path):
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        self.logger.info(
            f"Loaded DQN training checkpoint from {model_path}"
        )

    self.target_model = QNetwork().to(
        self.device
    )

    self.target_model.load_state_dict(
        self.model.state_dict()
    )

    self.target_model.eval()

    self.optimizer = optim.Adam(
        self.model.parameters(),
        lr=LR,
    )

    self.replay_buffer = deque(
        maxlen=BUFFER_SIZE
    )

    self.epsilon = EPSILON_START

    self.training_steps = 0

    self.round_reward = 0.0
    self.round_coins = 0

    self.reward_history = []
    self.coins_history = []


# Reward shaping
def add_custom_events(
    old_f,
    action,
    new_f,
    events,
):
    """
    Add the same custom v4 shaping events used by q_agent.
    """

    if old_f is None:
        return

    # Danger transitions
    if (
        old_f[F_URGENCY] > 0
        and new_f is not None
        and new_f[F_URGENCY] == 0
    ):
        events.append(
            ESCAPED_DANGER
        )

    if (
        old_f[F_URGENCY] == 0
        and new_f is not None
        and new_f[F_URGENCY] > 0
    ):
        events.append(
            ENTERED_DANGER
        )

    # Bomb quality
    if e.BOMB_DROPPED in events:

        opportunity = old_f[
            F_BOMB_OPPORTUNITY
        ]

        if opportunity == BOMB_NONE:

            events.append(
                SUICIDAL_BOMB
            )

        elif opportunity == BOMB_OPPONENT:

            events.append(
                SAFE_BOMB_NEAR_OPPONENT
            )

        elif opportunity in (
            BOMB_CRATES,
            BOMB_CRATES_MANY,
        ):

            events.append(
                SAFE_BOMB_NEAR_CRATES
            )

            if opportunity == BOMB_CRATES_MANY:

                events.append(
                    SAFE_BOMB_MULTI_CRATE
                )

        else:

            events.append(
                USELESS_BOMB
            )

    # Trap transitions
    if new_f is not None:

        if (
            old_f[F_MOBILITY] != MOBILITY_TRAP
            and new_f[F_MOBILITY] == MOBILITY_TRAP
        ):

            events.append(
                ENTERED_TRAP
            )

        elif (
            old_f[F_MOBILITY] == MOBILITY_TRAP
            and new_f[F_MOBILITY] != MOBILITY_TRAP
        ):

            events.append(
                LEFT_TRAP
            )

    # Still in danger
    if (
        old_f[F_URGENCY] > 0
        and new_f is not None
        and new_f[F_URGENCY] > 0
    ):

        events.append(
            STILL_IN_DANGER
        )

    # Objective movement
    moved_dir = next(
        (
            direction
            for event, direction
            in MOVED_EVENTS.items()
            if event in events
        ),
        None,
    )

    if (
        old_f[F_URGENCY] == URGENCY_SAFE
        and old_f[F_OBJECTIVE] != NO_DIR
        and moved_dir is not None
    ):

        if moved_dir == old_f[F_OBJECTIVE]:

            events.append(
                MOVED_TOWARD_OBJECTIVE
            )

        elif e.COIN_COLLECTED not in events:

            events.append(
                MOVED_AWAY_FROM_OBJECTIVE
            )


# Environment transition
def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events,
):
    """
    Called after every action.

    Creates one replay-buffer transition:

        state
        action
        reward
        next_state
        next_allowed_actions
        done
    """

    if old_game_state is None:
        return

    if self_action not in ACTIONS:
        return

    # Convert states to v4 features
    old_f = state_to_features(
        old_game_state
    )

    new_f = state_to_features(
        new_game_state
    )

    # Add q_agent-compatible shaping events
    add_custom_events(
        old_f,
        self_action,
        new_f,
        events,
    )

    # Calculate reward
    reward = reward_from_events(
        self,
        events,
    )

    self.round_reward += reward

    self.round_coins += events.count(
        e.COIN_COLLECTED
    )

    # Convert to numpy arrays
    state = np.asarray(
        old_f,
        dtype=np.float32,
    )

    next_state = np.asarray(
        new_f,
        dtype=np.float32,
    )

    action_idx = ACTIONS.index(
        self_action
    )

    next_allowed_actions = (
        get_allowed_next_actions(
            new_game_state
        )
    )

    # Environment transition is non-terminal here.
    done = False

    self.replay_buffer.append(
        (
            state,
            action_idx,
            reward,
            next_state,
            next_allowed_actions,
            done,
        )
    )

    # Perform one gradient update
    train_step(self)


# DQN update
def train_step(self):

    if len(self.replay_buffer) < MIN_BUFFER:
        return

    batch = random.sample(
        self.replay_buffer,
        BATCH_SIZE,
    )

    # States
    states = torch.tensor(
        np.stack(
            [transition[0] for transition in batch]
        ),
        dtype=torch.float32,
        device=self.device,
    )

    # Actions
    actions = torch.tensor(
        [
            transition[1]
            for transition in batch
        ],
        dtype=torch.long,
        device=self.device,
    )

    # Rewards
    rewards = torch.tensor(
        [
            transition[2]
            for transition in batch
        ],
        dtype=torch.float32,
        device=self.device,
    )

    # Next states
    next_states = torch.tensor(
        np.stack(
            [transition[3] for transition in batch]
        ),
        dtype=torch.float32,
        device=self.device,
    )

    # Next-state allowed actions
    next_allowed_actions = [
        transition[4]
        for transition in batch
    ]

    # Terminal flags
    dones = torch.tensor(
        [
            transition[5]
            for transition in batch
        ],
        dtype=torch.float32,
        device=self.device,
    )

    # Current Q(s, a)
    q_values = self.model(
        states
    )

    current_q = q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)


    with torch.no_grad():

        next_q_values = self.target_model(
            next_states
        )

        # Start with every action masked out.
        masked_next_q = torch.full_like(
            next_q_values,
            -1e9,
        )

        for i, allowed in enumerate(
            next_allowed_actions
        ):

            if allowed:

                masked_next_q[
                    i,
                    allowed,
                ] = next_q_values[
                    i,
                    allowed,
                ]

            else:

                # This should be extremely rare, but avoid creating an
                # invalid -1e9 target if something unexpected happens.
                masked_next_q[
                    i
                ] = next_q_values[
                    i
                ]

        max_next_q = masked_next_q.max(
            dim=1
        ).values

        target = rewards + (
            GAMMA
            * (1.0 - dones)
            * max_next_q
        )

    loss = nn.functional.smooth_l1_loss(
        current_q,
        target,
    )

    # Backpropagation
    self.optimizer.zero_grad()

    loss.backward()

    # Prevent occasional exploding gradients.
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        10.0,
    )

    self.optimizer.step()

    self.training_steps += 1

    # Target-network update
    if (
        self.training_steps
        % TARGET_UPDATE
        == 0
    ):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )



# End of round
def end_of_round(
    self,
    last_game_state,
    last_action,
    events,
):    
    """
    Handle the final reward and terminal transition.
    """

    # Final-round reward
    reward = reward_from_events(
        self,
        events,
    )

    self.round_reward += reward

    self.round_coins += events.count(
        e.COIN_COLLECTED
    )

    # Store terminal transition.
    #
    # There is no useful next-state action mask because done=True.
    if (
        last_game_state is not None
        and last_action in ACTIONS
    ):

        last_f = state_to_features(
            last_game_state
        )

        state = np.asarray(
            last_f,
            dtype=np.float32,
        )

        action_idx = ACTIONS.index(
            last_action
        )

        self.replay_buffer.append(
            (
                state,
                action_idx,
                reward,
                state,
                [],
                True,
            )
        )

    # One final training update
    if len(self.replay_buffer) >= MIN_BUFFER:

        train_step(self)

    # Decay exploration once per round
    self.epsilon = max(
        EPSILON_END,
        self.epsilon * EPSILON_DECAY,
    )

    # Save statistics
    self.reward_history.append(
        self.round_reward
    )

    self.coins_history.append(
        self.round_coins
    )

    n = len(
        self.reward_history
    )

    # Print statistics every 100 rounds
    if n % 100 == 0:

        print(
            f"[DQN] Round {n}: "
            f"avg reward (last 100) = "
            f"{np.mean(self.reward_history[-100:]):.1f}, "
            f"avg coins = "
            f"{np.mean(self.coins_history[-100:]):.2f}, "
            f"epsilon = "
            f"{self.epsilon:.3f}, "
            f"buffer = "
            f"{len(self.replay_buffer)}, "
            f"steps = "
            f"{self.training_steps}",
            flush=True,
        )

    # Reset round statistics
    self.round_reward = 0.0
    self.round_coins = 0

    torch.save(
        self.model.state_dict(),
        "dqn_model_v5.pt",
    )


# Reward calculation
def reward_from_events(
    self,
    events,
):
    """
    Convert game/custom events into the numerical reward used by DQN.
    """

    return sum(
        GAME_REWARDS.get(
            event,
            0.0,
        )
        for event in events
    )