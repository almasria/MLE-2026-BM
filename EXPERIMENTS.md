# Experiments

How every result in the report was produced, and how to reproduce it from
scratch. All commands run from the repository root. Windows PowerShell syntax
is shown for environment variables; on Linux/macOS use `VAR=value command`.

Conventions:
* Standard evaluation = 100 rounds against three `rule_based_agent`s on the
  `classic` scenario, via `scripts/evaluate.py`, which appends one row to
  `results/ledger.csv` and keeps the raw stats under `results/evaluations/`.
  Per-round scores are noisy (roughly +/-0.3 per round on a 100-round mean);
  compare means over seeds, never single 30-round runs.
* Score = coins + 5 x kills, the tournament's ranking metric.
* Training is resumable: the Q-table is saved every round. Set
  `Q_AGENT_EPS_START` when continuing a table so exploration does not reset
  to 100%.

Environment variables used for ablations (defaults in parentheses):

| variable | effect |
|---|---|
| `Q_AGENT_FEATURE_VERSION` (v5) | `v4` selects the 9-component feature set |
| `Q_AGENT_SYMMETRY` (1) | `0` disables D4 canonicalisation |
| `Q_AGENT_DEEP_PRUNING` (1) | `0` keeps only the one-step action mask |
| `Q_AGENT_THREAT_RADIUS` (4) | `0` disables the worst-case opponent bomb model |
| `Q_AGENT_ESCAPE_CLEARANCE` (2) | `0` disables the escape clearance requirement |
| `Q_AGENT_CONTESTED_COINS` (1) | `0` disables the contested-coin rule |
| `Q_AGENT_ENGAGE_RADIUS` (4) | reach within which an opponent counts as near |
| `Q_AGENT_GAMMA`, `Q_AGENT_ALPHA`, `Q_AGENT_EPS_DECAY` | learning hyperparameters |
| `Q_AGENT_MODEL_PATH` | evaluate or train a specific table without touching the default file |
| `Q_AGENT_DEATH_LOG` | `1` records the circumstances of every self-kill to `death_log.csv` |

---

## Part A. Training recipes

### A1. Full curriculum from scratch (the tournament model's origin)

```powershell
python train_tournament.py --fresh
```

Stages: coin-heaven 600 rounds, loot-crate 4000, classic vs 3 peaceful 2000,
vs 3 coin_collector 5000, vs 3 rule_based 5000. Each stage is evaluated
(30 rounds vs rule_based) and checkpointed to `results/checkpoints/`.
Roughly 8-10 hours. `--from-stage N` resumes; `--scale 0.02` is a
five-minute plumbing check; `--selfplay` appends a self-play stage (see B9).

### A2. Adaptation of an existing table

Used after changes to the objective cascade or the rewards, which keep the
Q-table keys valid but shift what some states mean:

```powershell
Copy-Item results\candidates\tournament.pkl agent_code\RL-Team\q_table_v5_sym.pkl
$env:Q_AGENT_EPS_START = "0.05"
python main.py play --agents RL-Team rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 2000 --no-gui
Remove-Item Env:Q_AGENT_EPS_START
```

### A3. Standard evaluation

```powershell
python scripts/evaluate.py --model results\candidates\tournament.pkl            # 100 rds vs rule_based
python scripts/evaluate.py --model results\candidates\tournament.pkl --seeds 3  # three seeds
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents mixed --rounds 50
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents solo-crates --rounds 30
```

### A4. Table audit

```powershell
python scripts\audit_table.py results\candidates\tournament.pkl
```

Reports escape discipline, greedy actions that would be fatal, and how many
states are well trained.

---

## Part B. Experiments completed

### B1. D4 symmetry ablation (feature set v1, coin-heaven)

Result: 80 raw feature tuples collapse to 18 orbits (6 reachable states vs
32 without symmetry). Training curves under identical exploration were
indistinguishable (rounds-to-45-coins 357 +/- 9 vs 357 +/- 3); the gain was
table size and convergence reliability, not speed.

Reproduce using the historical features-v1 tag. A separate worktree keeps the current RL-Team checkout unchanged.

```powershell
git worktree add ..\mle-features-v1 features-v1
Set-Location ..\mle-features-v1
python run_symmetry_ablation.py --seeds 3 --rounds 1000 --eval-rounds 20
python analyze_symmetry.py
```
At this tag, the experiment scripts are in the repository root rather than the current experiments/ directory

Artifacts: `results/ablation_summary.csv`, `results/symmetry_ablation.png`,
`results/sym{0,1}_seed{0,1,2}_eval.json`.

Still to do at the current feature set (cheap, strong figure):

```powershell
$env:Q_AGENT_SYMMETRY = "0"
python train_tournament.py --fresh      # writes q_table_v5_plain.pkl
Remove-Item Env:Q_AGENT_SYMMETRY
python scripts/evaluate.py --model agent_code\RL-Team\q_table_v5_plain.pkl --seeds 2
```

### B2. Feature set size: 22 components (v3) vs 9 (v4)

Result at equal training (600 rounds loot-crate, seed 0): v3 had 3568
states with 8% well trained and 35% survival; v4 had 172 states with 81%
well trained and 80% survival. A tabular learner cannot afford continuous
or high-cardinality components: every component multiplies the state space
and divides the visits per state.

Reproduce: tag `w3-milestone` holds v3; `experiments/featuresv3.py` and
`experiments/feature_ablation.py` are the archived code. The v4 side is the
current code with `Q_AGENT_FEATURE_VERSION=v4`.

### B3. Reward economy (crate rewards vs coin rewards)

Observed: the agent walked past coins to bomb crates. Cause: a triple-crate
bomb earned +22 in shaping versus +10 for a real coin, while a crate in the
tournament scenario is worth ~0.07 points (9 coins under ~130 crates).
Change: `CRATE_DESTROYED` 5 -> 1, bomb-placement bonuses 4/3 -> 1.5/1.5,
`COIN_FOUND` 3 -> 5. Effect on the 100-round benchmark: coins per round
2.4 -> 2.66; the agent won 406 vs 318 while destroying fewer crates.

Reproduce: edit the values in `agent_code/RL-Team/config.py`, run A2, then A3.
Artifact: `results/eval_econ.json`.

### B4. Survivability pruning (full-depth action mask)

Observed: 0% survival on loot-crate with a one-step mask despite a correct
escape feature; self-kills came from entering tiles that were safe now but
dead ends one step later. Change: prune every action after which no
survival line exists (`feature_core.survivable_actions`). Effect: solo
survival 50% -> 100% with more crates cleared (28 -> 49 per round).

Reproduce (solo, no retraining needed):

```powershell
$env:Q_AGENT_DEEP_PRUNING = "0"
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents solo-crates --rounds 30 --tag no-deep-pruning
Remove-Item Env:Q_AGENT_DEEP_PRUNING
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents solo-crates --rounds 30 --tag deep-pruning
```

### B5. Worst-case opponent bomb model

Observed via `Q_AGENT_DEATH_LOG=1`: 24 of 24 self-kills against rule_based
had an armed opponent within two tiles, another bomb active, and no escape
left; rule_based bombs whenever it touches an agent. Change: armed
opponents within `THREAT_RADIUS` are assumed to bomb immediately; bomb
drops and moves must survive that. Effect with the same table: score 113
-> 147, suicides 14 -> 10 (30 rounds).

Reproduce:

```powershell
$env:Q_AGENT_THREAT_RADIUS = "0"
python scripts/evaluate.py --model results\candidates\tournament.pkl --tag no-threat-model
Remove-Item Env:Q_AGENT_THREAT_RADIUS
python scripts/evaluate.py --model results\candidates\tournament.pkl --tag threat-model
```

Death log for the diagnosis:

```powershell
$env:Q_AGENT_DEATH_LOG = "1"
python main.py play --agents RL-Team rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 60 --no-gui --continue-without-training
Remove-Item Env:Q_AGENT_DEATH_LOG
# columns of agent_code/RL-Team/death_log.csv: step, nearest opponent distance,
# other active bombs, urgency, safe_dir, mobility, last action
```

### B6. Endgame hunting

In the tournament scenario only 9 coins exist; once collected, crates are
worthless and only kills score. Coins collected are inferred from visible
scores (`feature_core.hidden_coins_remaining`), after which the objective
switches to hunting. Crates remain a target while they block the path to
the opponents.

### B7. Reward pump removed (danger shaping)

Observed: the agent parked in a corner dropping bombs and escaping. Cause:
escape rewards summed to ~+12 per self-created danger cycle. Change:
removed action-dependent escape rewards; entering danger costs -3 even when
self-inflicted, leaving pays +3, each step inside costs -0.75, so any cycle
nets <= 0 unless crates or opponents are hit. Effect: coin-heaven bombs
0.75 -> 0.00 per round, loot-crate coins 33.8 -> 40.2 per round.

Reproduce: tag `w3-milestone` (before) vs current `config.py`; evaluate with
`--opponents solo-coins` and `--opponents solo-crates`.

### B8. Gamma sweep

Fresh full curriculum per (value, seed), 100-round evaluation:
gamma=0.9: 411, 470 (mean 440, suicides 12/13);
gamma=0.95: 390, 378 (mean 384, suicides 33/22). Default kept at 0.9.

```powershell
python run_sweep.py --param Q_AGENT_GAMMA --values 0.9 0.95 --seeds 2
```

Artifacts: `results/sweep_summary.csv`, `results/sweep_models/`,
`results/sweep_logs/`. Note: the archived `Q_AGENT_GAMMA=0.9_s0.pkl` has
510 states and was overwritten by an interrupted re-run; its sweep-time
result (411) is not reproducible from that file.

### B9. Self-play degradation

3000 rounds against identical copies after the rule_based stage lowered the
rule_based evaluation from 116 to 77 (30 rounds); copies never bomb when
adjacent, so the policy tuned to rule_based drifts. Self-play is opt-in.

```powershell
python train_tournament.py --from-stage 5 --selfplay
```

Artifact: `results/eval_6-6-selfplay.json`, checkpoint
`results/checkpoints/q_table_after_6-6-selfplay.pkl`.

### B10. Two-tile oscillation

Measured 21-25% of steps in two-tile loops with position traces. Cause
traced to the objective cascade: standing on the best bomb spot returned
"no objective" and fell through to the hunt direction; leaving the spot
cost nothing while walking to it paid +1. Fixes in `feature_core` and
`train.py`. Effect on one base table: loops 21-25% -> 15.7%, 25-round score
46-55 -> 84.

Reproduce the measurement: add a position trace to `act` (see the notes in
`doc/CLEANUP_RECORD.md`, section 3b) and count A,B,A,B sequences.

### B11. Contested coins and escape clearance

Observed in play: the agent chased a far coin surrounded by opponents while
standing next to crates, and escaped its own bomb toward an opponent who
sealed the corridor. Changes: contested coins rank below crate work;
escapes prefer safe tiles with clearance from armed opponents, and a bomb
near armed opponents requires such an escape. Effect on one base table
(30 rounds): suicides 4/25 -> 2/30, coins 49 -> 63, score flat.

```powershell
$env:Q_AGENT_CONTESTED_COINS = "0"; $env:Q_AGENT_ESCAPE_CLEARANCE = "0"
python scripts/evaluate.py --model results\candidates\tournament.pkl --tag without-b11
Remove-Item Env:Q_AGENT_CONTESTED_COINS; Remove-Item Env:Q_AGENT_ESCAPE_CLEARANCE
python scripts/evaluate.py --model results\candidates\tournament.pkl --tag with-b11
```

### B12. Candidate comparison and the tournament model

Four tables under the standard evaluation (100 rounds vs rule_based):

| table | score | /round | vs best opp | kills | suicides |
|---|---|---|---|---|---|
| pipeline stage 5 (A1) | 483 | 4.83 | +64% | 42 | 24 |
| gamma 0.9 seed 1 (B8) | 460 | 4.60 | +51% | 36 | 17 |
| pipeline stage 4 | 406 | 4.06 | +29% | 31 | 35 |
| gamma 0.9 seed 0 (partial file) | 227 | 2.27 | -44% | 13 | 16 |

Stage 5 adapted under the B11 code (A2 with 2000 + 5000 rounds):
**479, 4.79/round, +62% vs best opponent, 42 kills, 18 suicides (82%
survival)** -- `results/candidates/tournament.pkl` (1199 states). Chosen
over the unadapted stage 5 (483, 24 suicides): equal score within noise,
fewer deaths.

### B13. Adaptation length (convergence check)

Continuing the adapted table for a further 10,000 rounds at the epsilon
floor gave 416 (4.16/round, 31 kills, 14 suicides) against the 5,000-round
table's 479. Beyond a few thousand adaptation rounds the table no longer
improves and its greedy policy drifts; more rounds are not a lever once
the reachable states are well trained (see `scripts/audit_table.py`
coverage). Artifact: `results/candidates/adapted10000_eval100.json`.

Artifacts: `results/candidates/*_eval100.json`, `results/candidates/LEDGER.md`.

---

## Part C. Experiments still to do

Ordered by value for the report; each is one or two commands.

### C1. Multi-seed confirmation of the tournament model (required)

```powershell
python scripts/evaluate.py --model results\candidates\tournament.pkl --seeds 3
```

Report mean and spread of the three per-round scores.

### C2. Robustness against a mixed field (required)

The tournament field is other teams' agents, not three rule_based copies.

```powershell
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents mixed --rounds 100
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents collector --rounds 100
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents peaceful --rounds 100
```

### C3. Per-task performance (matches the report's task 1-4 structure)

```powershell
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents solo-coins --rounds 30
python scripts/evaluate.py --model results\candidates\tournament.pkl --opponents solo-crates --rounds 30
```

Plus C2 for tasks 3 and 4.

### C4. Symmetry ablation at the current feature set (see B1, second block)

### C5. Feature set v4 vs v5 at equal training budget

```powershell
$env:Q_AGENT_FEATURE_VERSION = "v4"
python train_tournament.py --fresh          # writes q_table_v4_sym.pkl
python scripts/evaluate.py --model agent_code\RL-Team\q_table_v4_sym.pkl --seeds 2
Remove-Item Env:Q_AGENT_FEATURE_VERSION
```

Compare with the v5 pipeline result (B12, stage 5).

### C6. Design ablations on the final table (no retraining)

Each of B4, B5 and B11 has a reproduce block above; run them against
`tournament.pkl` and tabulate score, coins, kills and suicides per switch.

### C7. Submission test

Build the Docker image from the provided `Dockerfile`, copy the agent folder
in, run one game with the tournament table, check `logs/game.log` and the
agent log for errors and decision time. Then upload for the pre-run.

### C8. Second model (DQN)

`agent_code/dqn_agent` agent_code/dqn_agent/ is tracked and includes the v4 and v5 model files. It still needs evaluation evidence produced with the same scenario, opponents, round count and seed policy as RL-Team before the models can be compared fairly.
The current scripts/evaluate.py is specific to RL-Team. Extend it with an agent argument, or document an equivalent DQN evaluation command, before reporting a direct comparison.
