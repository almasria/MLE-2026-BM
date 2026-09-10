# Two-Model Project Plan — RL for Bomberman (ML Essentials, SS 2026)

Deadlines (from the project sheet):
- **Wed 17.09, 21:00** — submission test deadline (Docker pre-run on tournament machines)
- **Mon 21.09, 21:00** — agent code (zip of your best agent's `agent_code` subfolder → MaMPF)
- **Mon 28.09, 21:00** — report (~4000 words/member, public repo URL inside)

Today: Wed 13.08 → **5.5 weeks to code, 6.5 to report.**

---

## The two models

| | Model A: `q_agent` | Model B: `dqn_agent` |
|---|---|---|
| Approach | Tabular Q-learning + D4 symmetry canonicalization (upgrade path: linear approximation) | Deep Q-Network (PyTorch): replay buffer + target network |
| Role | Safe baseline; likely tournament submission | Ambitious comparison; submitted only if it clearly wins |
| Lecture-technique requirement | ✔ covers it | — |
| Risk | Low; converges in minutes on CPU | High; sheet warns DQNs often unconverged by deadline |

**Both models consume the SAME feature vector and the SAME reward shaping.**
This is the single most important design decision:
- It makes the comparison scientific (isolates the variable: function approximator).
- It halves the work (features/rewards are built once, by the whole team).
- It satisfies the no-split-labor rule: nobody "owns" a model; people own components.

## Team split by COMPONENT (not by model — the sheet forbids that)

- **Features & symmetry**: `features.py` used by both agents; BFS helpers; canonicalization.
- **Rewards & events**: shared `reward_from_events` + custom events; tuning experiments.
- **Training infra & experiments**: run scripts, logging, learning-curve plots, evaluation
  protocol (see below), Docker submission test.
- **DQN specifics**: network architecture, replay/target-net hyperparameters.
Everyone reviews everything; rotate pairing weekly. Assign report sections early
(each section needs a named main author for grading).

## Evaluation protocol (define ONCE, in week 1, use everywhere)

- Metrics: coins/round, kills, deaths (self vs. opponent), win rate, score/round,
  steps survived, decision time per step (must stay ≪ 0.5 s on CPU!).
- Standard benchmark: 100 rounds, fixed seeds, vs. 3 × rule_based_agent (task 4),
  plus per-task benchmarks (task 1: coin-heaven; task 2: loot-crate solo;
  task 3: vs peaceful + coin_collector).
- Every experiment logged to CSV → one plotting script → figures for the report.
- Head-to-head: q_agent vs dqn_agent in the same match = free report data.

---

## Week-by-week

**W1 — Aug 13–17 | Foundations sprint + skeleton repo**
- Compressed theory: Sutton & Barto Ch. 3, 6 (MDPs, TD, Q-learning) + Silver L1–5
  in parallel with coding. Ch. 9–10 + DQN reading can trail into W2.
- Set up public repo, team name, MaMPF group. Copy framework, both agent folders.
- Get the coin-collector baseline (already working) renamed → `q_agent`; everyone
  runs training locally once. Define the evaluation protocol + logging format.
- Milestone: both agents run without crashing; q_agent solves coin-heaven.

**W2 — Aug 18–24 | Task 1 complete for BOTH models + symmetry**
- Shared `features.py` v1 (coin direction + walkability). DQN learns coin-heaven.
- Implement D4 canonicalization for q_agent; measure the gain (rounds-to-convergence
  with vs. without) → first report experiment.
- Milestone: both models ≥ 45/50 coins on coin-heaven benchmark; symmetry plot done.

**W3 — Aug 25–31 | Task 2: crates, bombs, survival**
- Features v2: danger flag, escape direction, bomb-safety flag, crates-in-range.
- Rewards v2: CRATE_DESTROYED, COIN_FOUND, KILLED_SELF ≪ 0, custom ESCAPED_DANGER.
- Curriculum: warm-start from task-1 weights → loot-crate scenario.
- Milestone: both agents clear loot-crate without self-kills in ≥ 90% of rounds.

**W4 — Sep 1–7 | Tasks 3–4: opponents**
- Features v3: nearest-opponent direction, opponent-in-blast-range flag.
- Train vs peaceful → coin_collector → rule_based; start self-play experiments.
- Hyperparameter sweeps (α, γ, ε-schedule; DQN: lr, buffer, target-sync, net size).
- Milestone: beat peaceful + coin_collector consistently; competitive rounds vs
  rule_based_agent.

**W5 — Sep 8–14 | Beat the rule_based_agent + freeze features**
- Focused iteration on the task-4 benchmark ("you must beat rule_based_agent to
  have any chance in the tournament").
- Decide the tournament model from benchmark data. Feature freeze Fri 12.09.
- Full 100-round evaluations of all variants → report figures.
- Docker test locally (build image, run agent inside, check decision-time margin).

**W6 — Sep 15–21 | Submission week**
- **By Wed 17.09**: upload to MaMPF submission test; fix whatever the pre-run reports.
- Continue training the frozen architecture (more rounds is free performance).
- Final check: relative paths only, trained parameters inside the agent folder,
  requirements.txt (torch!) if dqn_agent is submitted.
- **Mon 21.09**: submit zip.

**W7 — Sep 22–28 | Report**
- Structure per sheet §9; most effort into §6 Experiments & Results.
- Ready-made experiment narrative: symmetry gain, tabular vs DQN sample efficiency,
  reward-shaping ablations, hyperparameter sweeps, head-to-head, per-task curves.
- Everyone writes their named sections; cross-review; **Mon 28.09**: submit PDF + repo URL.

## Standing risks
- DQN not converged by W5 → submit q_agent (this is the plan working, not failing).
- Decision-time limit: profile `act` early; a Q-table lookup is ~µs, a small MLP on
  CPU is ~ms — both fine, but BFS feature code must stay efficient (no per-step
  full-board floods beyond what's needed).
- Rules may change until 7 days before deadline → re-run benchmarks after any
  framework `git pull`.
- AI-use rule: drafts must be refined in your own style and the main work must
  clearly be yours — treat all starter code as scaffolding to rewrite and extend,
  and disclose AI assistance per the sheet.
