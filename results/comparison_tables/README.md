# Comparison Table Artifacts

These files come from `experiments/baseline_comparison.py`.

## Evidence Basis

- seeded synthetic calibration
- adversarial-simulation comparisons for selected OLYMPUS modules
- internal method ranking inside a controlled toy setup

## Experiments Included

### Module 2 Synthetic Malware Detection Calibration
- Module: Module 2 / Malware Detection
- Evidence type: synthetic benchmark-style calibration
- Primary metric: `f1`
- Reference method: `Random Forest (baseline)`
Not claimed:
- real-world malware-detection superiority
- EMBER or VirusShare benchmark leadership
- deployment-ready SOC false-positive behavior

### Module 7 Social-Engineering Detection Calibration
- Module: Module 7 / Social Engineering
- Evidence type: synthetic adversarial-email calibration
- Primary metric: `f1`
- Reference method: `SpamAssassin (baseline)`
Not claimed:
- production email-security accuracy on live traffic
- enterprise deployment readiness
- real phishing-corpus superiority

### Module 9 LLM Defense Calibration
- Module: Module 9 / LLM Defense
- Evidence type: synthetic adversarial prompt-defense calibration
- Primary metric: `f1`
- Reference method: `Pattern-matching only`
Not claimed:
- real deployment safety across public jailbreak benchmarks
- operational LLM-guardrail superiority
- production moderation behavior on user traffic

### Module 6 TITAN Evolution Calibration
- Module: Module 6 / TITAN Co-Evolution
- Evidence type: synthetic co-evolutionary simulation
- Primary metric: `best_fitness`
- Reference method: `Standard-GA`
Not claimed:
- real-world autonomous red-team/blue-team dominance
- validated search superiority on operational cyber ranges
- field-ready attack automation performance

## Reading Guidance

Use `comparison_results.json` as the canonical machine-readable source. The LaTeX tables
are derived summaries intended for manuscript/report insertion, not a replacement for the
seed-level raw rows preserved in the same JSON file.
