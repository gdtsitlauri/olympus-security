# OLYMPUS Results Notes

The files under `results/` are synthetic or simulation-first research artifacts unless a
future release explicitly adds external benchmark raw outputs and labels them separately.

## What These Files Are For

- internal comparative sanity checks
- attacker/defender calibration under seeded synthetic scenarios
- export-pipeline validation and regression tracking
- preserving tables and JSON summaries that match the paper and README

## What These Files Do Not Prove

- real-world SOC or enterprise deployment performance
- public-benchmark leadership on malware, phishing, or jailbreak datasets
- operational superiority over fielded commercial tools

## Subdirectories

- `comparison_tables/`: seeded comparison summaries for Modules 2, 6, 7, and 9
- `ablation/`: TITAN component-ablation summaries

Inspect the per-folder READMEs and the `metadata` blocks inside the JSON exports before
citing any number from this directory.
