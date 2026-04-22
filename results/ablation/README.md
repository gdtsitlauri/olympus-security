# TITAN Ablation Artifacts

These files come from `experiments/ablation_study.py`.

## Evidence Basis

- seeded synthetic co-evolutionary simulation
- component-removal analysis for TITAN
- internal algorithm attribution under a shared scoring model

## What The Metrics Mean

- `best_attack_fitness` (higher is better): Synthetic attacker-side objective value under the shared TITAN scoring model.
- `best_defense_fitness` (higher is better): Synthetic defender-side objective value under the shared TITAN scoring model.
- `convergence_generation` (lower is better): Generation index at which the seeded run reached its best recorded score.

## What Is Not Claimed

- field-validated attacker/defender superiority
- operational cyber-range performance
- hardware- or deployment-level optimization claims

Use `ablation_results.json` as the canonical machine-readable source. The LaTeX table
is a derived manuscript summary and should be read together with the seed-level raw runs
and significance metadata in the JSON export.
