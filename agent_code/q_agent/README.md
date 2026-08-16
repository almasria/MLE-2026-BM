# q_agent — verified stage-1 starter for bomberman_rl

A minimal tabular Q-learning agent for the Heidelberg ML Essentials framework
(github.com/ukoethe/bomberman_rl), tested against the actual framework code.

## Verified results (on the real framework, July 2026 master branch)

- Scenario: `coin-heaven` (50 coins, no crates, no opponents)
- Training: 1300 rounds, ~4 minutes on CPU, no GPU needed
- Learning curve: avg coins/round 23 → 32 → 40 → **49.9** as ε decayed
- Greedy evaluation: **48.8 / 50 coins per round** over 10 rounds
- Entire learned model: a Q-table with **32 states** (2.5 KB pickle)

The included `q_table.pkl` is the trained model, so the agent plays well
immediately — but delete it and retrain yourself; watching the learning
curve is the whole point.

## Install & run

1. Clone the framework: `git clone https://github.com/ukoethe/bomberman_rl`
2. Copy this folder into `bomberman_rl/agent_code/`
3. `pip install pygame tqdm numpy`
4. Train (fast, headless):
   `python main.py play --agents q_agent --train 1 --scenario coin-heaven --n-rounds 1000 --no-gui`
5. Watch it play:
   `python main.py play --agents q_agent --scenario coin-heaven`
6. Progress is logged to `agent_code/q_agent/logs/` every 100 rounds
   (avg reward, avg coins, epsilon, Q-table size).

## How it works

- **Features** (`callbacks.py: state_to_features`): a 5-tuple —
  BFS direction to the nearest coin (up/right/down/left/none) + walkability
  of the 4 neighboring tiles. Tiny state space, fully sufficient for stage 1.
- **Learning** (`train.py`): textbook tabular Q-learning with ε-greedy
  exploration, ε decaying 1.0 → 0.05 over ~2000 rounds.
- **Reward shaping**: +10 coin collected, ±1 for moving toward/away from the
  nearest coin (a custom event computed from the BFS feature), small penalties
  for invalid moves and waiting.

## A bug you should learn from (we hit it during testing)

The first version only explored during training. In greedy play mode the
policy became fully deterministic — and a deterministic policy in a
deterministic world can bounce between two states forever. Performance
collapsed from ~50 to ~4.6 coins/round. Fix: keep a small ε (10%) even in
play mode to break loops. Lesson: **evaluate your agent greedily, not just
during training** — training metrics with high ε can hide policy defects.

## Extending to stage 2 (loot-crate) and 3 (classic)

The pipeline stays identical; you grow the feature tuple and reward table:

1. Widen the action set: use `ACTIONS` (incl. `BOMB`, `WAIT`) instead of
   `STAGE1_ACTIONS` in both files.
2. New features (reuse `bfs_direction_to_nearest` with different targets):
   - direction to nearest crate; number of crates a bomb here would hit
   - danger flag: in blast radius / on explosion tile (from `game_state['bombs']`
     and `game_state['explosion_map']`)
   - direction to nearest safe tile when in danger
   - "bomb here has an escape route" flag  ← prevents KILLED_SELF
3. New rewards in `GAME_REWARDS` (already sketched in train.py):
   CRATE_DESTROYED, COIN_FOUND, KILLED_SELF (big negative!), GOT_KILLED,
   KILLED_OPPONENT, SURVIVED_ROUND — plus custom ESCAPED_DANGER etc.
4. Train in curriculum order: keep the coin-heaven Q-values as a warm start,
   then `--scenario loot-crate`, then `--scenario classic` vs
   `rule_based_agent`.
5. When the tuple state space gets large, switch the dict Q-table to linear
   function approximation (weights per feature) — Sutton & Barto Ch. 9–10.
