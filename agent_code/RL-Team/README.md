# RL-Team

A tabular Q-learning agent for Bomberman (agent folder `RL-Team`). The game state is reduced to a
tuple of 10 small integers, canonicalised under the board's D4 symmetry, and
looked up in a dictionary Q-table. A safety mask removes actions after which
no survival path exists; the learned policy chooses among the rest.

## Files

| file | role |
|---|---|
| `callbacks.py` | `setup`, `act`: model loading, action mask, epsilon-greedy choice |
| `train.py` | `setup_training`, `game_events_occurred`, `end_of_round`: reward shaping, Q update, checkpointing |
| `config.py` | hyperparameters and reward table (all overridable via environment variables) |
| `features.py` | dispatcher: selects the feature version, canonicaliser, layout and model name |
| `feature_core.py` | shared feature computations: danger timing, escape search, bomb ranking, mobility, engagement |
| `featuresv4.py` | 9-component feature set (ablation baseline) |
| `featuresv5.py` | 10-component feature set (default): v4 plus engagement |
| `symmetry.py` | D4 group and tuple canonicalisation |
| `q_table_v5_sym.pkl` | trained model; must ship with the agent |

## Feature tuple (v5)

| idx | component | values |
|---|---|---|
| 0 | objective direction | 0-3 direction, 4 none |
| 1-4 | neighbour enterable | 0/1 per direction |
| 5 | urgency | 0 safe, 1 lethal in 2+ steps, 2 lethal within 1 |
| 6 | escape direction | 0-3, 4 none |
| 7 | bomb opportunity | 0 none, 2 crates, 3 many crates, 4 opponent |
| 8 | mobility | 0 trapped, 1 tight, 2 open |
| 9 | engagement | 0 no opponent near, 1 advantage, 2 no advantage |

The objective (index 0) is chosen by a priority cascade: a coin within a few
steps, then a vulnerable opponent during an advantage window, then any
uncontested coin, then the best crate-bombing spot, then a contested coin,
then the nearest opponent once coins and useful crates are gone. A coin is
contested when an opponent can reach it first, an armed opponent stands next
to it, or several opponents crowd it.

## Run

Train through the full curriculum (solo boards first, then opponents):

```
python train_tournament.py
```

Or train a single stage directly:

```
python main.py play --agents RL-Team --train 1 --scenario coin-heaven --n-rounds 600 --no-gui
python main.py play --agents RL-Team --train 1 --scenario loot-crate --n-rounds 3000 --no-gui
python main.py play --agents RL-Team rule_based_agent rule_based_agent rule_based_agent \
       --train 1 --scenario classic --n-rounds 5000 --no-gui
```

Evaluate and watch:

```
python main.py play --agents RL-Team rule_based_agent rule_based_agent rule_based_agent \
       --scenario classic --n-rounds 100 --no-gui --save-stats results/eval.json
python main.py play --agents RL-Team rule_based_agent rule_based_agent rule_based_agent \
       --scenario classic
```

The Q-table is saved every round, so an interrupted run resumes from its last
state. Per-round metrics accumulate in `training_history.csv`; the framework's
own log in `logs/` is overwritten each run.

## How it works

State is `state_to_features(game_state)` in the active feature module, built
from the shared helpers in `feature_core.py`. Danger is time-aware: bomb
countdowns and the explosion map are turned into a per-tile "lethal from / until"
window, and the escape search runs over (tile, time) pairs so it only routes
through tiles that can be cleared before they detonate. Symmetry reduces the
table by up to 8x: all eight rotations and reflections of a state share one
row via `canonicalize`, and the chosen action is mapped back through the
inverse group element.

Learning is standard tabular Q-learning with an epsilon-greedy behaviour
policy. The base rewards (coin, kill, death) are sparse, so `config.py` adds
shaping events; these depend only on the state before and after a step, and
the danger terms are symmetric so no cycle can farm reward. Escape execution
is enforced by the action mask rather than by rewards.

## Configuration

Every hyperparameter reads from an environment variable, so sweeps and
ablations need no code changes. The common ones:

| variable | default | purpose |
|---|---|---|
| `Q_AGENT_FEATURE_VERSION` | `v5` | `v4` or `v5` |
| `Q_AGENT_MODEL_PATH` | `q_table_<version>_sym.pkl` | model file (relative to this folder) |
| `Q_AGENT_SYMMETRY` | `1` | `0` disables canonicalisation |
| `Q_AGENT_ALPHA`, `Q_AGENT_GAMMA` | `0.1`, `0.9` | learning rate, discount |
| `Q_AGENT_EPS_START/END/DECAY` | `1.0`, `0.05`, `0.995` | exploration schedule |
| `Q_AGENT_PLAY_EPS` | `0.02` | exploration in play mode |
| `Q_AGENT_SEED` | unset | seeds the agent's random choices |
| `Q_AGENT_THREAT_RADIUS` | `4` | armed opponents within this range are assumed to bomb now |
| `Q_AGENT_ESCAPE_CLEARANCE` | `2` | required distance between an escape's safe tile and armed opponents |

A small play-mode epsilon is deliberate: a fully deterministic policy in a
deterministic world can bounce between two tiles forever, and a little noise
breaks such loops. Evaluate greedily rather than trusting training metrics,
which are taken under high exploration and can hide policy defects.

## Extending

`feature_core.py` holds the reusable pieces (BFS with distance, blast and
lethality timing, the temporal escape search, mobility, engagement). A new
feature version is a thin module that assembles a tuple from a `Perception`
object plus an entry in `symmetry.LAYOUTS` describing which components are
directions. Keep the tuple small: the state space is the product of each
component's range, and a tabular table needs every reachable state visited
many times to converge.
