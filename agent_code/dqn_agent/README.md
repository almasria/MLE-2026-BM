# dqn_agent — Model B: Deep Q-Network starter (verified)

The second model for the two-model requirement. Pairs with the tabular
q_agent (formerly coin_collector_agent — RENAME IT: the official framework
ships its own `coin_collector_agent` as the task-3 opponent, so the old
name collides).

## Verified results (real framework, Aug 2026 master, CPU only)

- Scenario: coin-heaven, 400 training rounds (~4 min on CPU)
- Learning curve: avg coins 37.8 (round 100) -> 50.0 (round 200 onward)
- Greedy evaluation: **50.0 / 50 coins per round** over 10 rounds
- Inference: ~0.04 ms/step — vs the tournament's 0.5 s/step limit
- Reached 50/50 in FEWER rounds than the tabular agent: the network
  generalizes across similar feature vectors instead of learning each
  state separately. (First data point for your model-comparison section.)

`dqn_model.pt` contains the trained weights. Delete it to retrain from
scratch. `requirements.txt` (torch) must ship with the agent if you
submit this model to the tournament.

## Design: the controlled comparison

Both models consume the SAME features (one-hot coin direction + 4
walkability flags = 9 dims) and the SAME reward shaping. The only
variable is the function approximator:

| | q_agent | dqn_agent |
|---|---|---|
| Q-function | table: dict[features] -> 4 values | MLP 9 -> 64 -> 64 -> 4 |
| Update | tabular TD | Adam on Huber TD loss |
| Stability tricks | none needed | replay buffer + target network |

DQN's two stabilizers (Mnih et al. 2015), both in train.py:
1. **Experience replay** — train on random mini-batches from a 50k buffer;
   breaks temporal correlation and reuses experience.
2. **Target network** — TD targets from a frozen copy synced every 1000
   steps; prevents the target-chasing feedback loop.

Both agents keep a small epsilon even in play mode — the deterministic-
policy loop bug we verified with the tabular agent applies to any
deterministic policy, including an argmax over network outputs.

## Install & run

1. Copy this folder into `bomberman_rl/agent_code/`
2. `pip install torch` (plus pygame, tqdm for the framework)
3. Train:  `python main.py play --agents dqn_agent --train 1 --scenario coin-heaven --n-rounds 400 --no-gui`
4. Watch:  `python main.py play --agents dqn_agent --scenario coin-heaven`
5. Logs in `agent_code/dqn_agent/logs/` every 100 rounds.

## Extending (weeks 3-4 of the plan)

- Grow `state_to_features` in BOTH agents together (danger flag, escape
  direction, bomb-safety, crate/opponent directions) — then FEATURE_DIM
  and the net's input layer here; the rest is untouched.
- Widen the action set to all 6 actions (STAGE1_ACTIONS -> ACTIONS) when
  bombs enter the curriculum.
- Symmetry for a network = data augmentation: push all 8 transformed
  copies of each transition into the replay buffer (transform features
  AND action index consistently). This is the DQN counterpart of the
  q_agent's canonicalization — and an ablation experiment for the report.
- Hyperparameters to sweep for the report: lr, buffer size, target-sync
  interval, epsilon decay, network width/depth.

## Honest caveats

- Stage 1 is where DQN looks easy. The sheet's warning about unconverged
  networks applies to stages 2-4, where the feature space and action set
  grow — budget real time for tuning, and keep q_agent as the fallback.
- Training is nondeterministic (init, replay sampling): rerun key
  experiments with 3+ seeds and report mean ± spread, not single runs.
- Per the sheet's AI-use rule: treat this as scaffolding to rewrite in
  your own style and extend substantially — and disclose AI assistance.
