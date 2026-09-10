# Candidate tables

All rows: 100 rounds vs three rule_based_agent, classic scenario, one seed,
evaluated with the current feature code (contested coins, escape clearance).

| file | origin | states | score | /round | best opp | coins | kills | suicides | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `tournament.pkl` | stage 5 + 2000 + 5000 adaptation rounds vs rule_based | 1199 | 479 | 4.79 | 295 | 269 | 42 | 18 | **tournament model** |
| (not kept) | tournament.pkl + 10000 more adaptation rounds | - | 416 | 4.16 | 299 | 261 | 31 | 14 | worse: over-adaptation, see EXPERIMENTS.md B13 |
| `pipeline_stage5.pkl` | full curriculum, after the rule_based stage | 1117 | 483 | 4.83 | 294 | 273 | 42 | 24 | runner-up (more deaths) |
| `gamma09_s1.pkl` | gamma sweep, 0.9, seed 1 | 1025 | 460 | 4.60 | 304 | 280 | 36 | 17 | strong alternative |
| `pipeline_stage4.pkl` | full curriculum, after the coin_collector stage | 905 | 406 | 4.06 | 315 | 251 | 31 | 35 | 30-round result (5.00) did not hold |
| `gamma09_s0.pkl` | gamma sweep, 0.9, seed 0 | 510 | 227 | 2.27 | 402 | 162 | 13 | 16 | partial file (interrupted re-run); sweep-time result 411 not reproducible |

Raw stats: `*_eval100.json` in this folder. New rows: `scripts/evaluate.py`
appends to `results/ledger.csv`.
