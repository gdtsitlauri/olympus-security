# OLYMPUS-SECURITY

**AI-native cyber offense/defense research platform for synthetic adversarial experimentation, adaptive security reasoning, and co-evolutionary attacker/defender studies.**

**Author:** George David Tsitlauri  
**Email:** gdtsitlauri@gmail.com  
**Website:** [gdtsitlauri.dev](https://gdtsitlauri.dev)  
**GitHub:** [github.com/gdtsitlauri](https://github.com/gdtsitlauri)

| Item | Status |
| --- | --- |
| Core platform code | Implemented |
| FastAPI API / CLI / dashboard | Implemented |
| Synthetic multi-module calibration exports | Implemented |
| TITAN co-evolution and ablation pipeline | Implemented |
| Real operational benchmark campaigns | Not yet committed |
| Safe public claim | High-end synthetic attacker/defender research platform |

## Overview

OLYMPUS-SECURITY is not a generic public-dataset IDS repository. It is a modular cyber research environment for building, orchestrating, and evaluating AI-native offense/defense components under controlled synthetic and adversarial conditions. Its core value is the combination of:

- a real multi-module cyber codebase,
- a shared orchestration and knowledge layer,
- explicit attacker/defender experimentation pathways,
- a co-evolutionary optimization engine called **TITAN**, and
- reproducible result exports that now label evidence type, metric semantics, and non-claims directly in the saved artifacts.

This makes OLYMPUS suitable for research on adaptive cyber reasoning, red/blue strategy search, synthetic adversarial calibration, and internal comparative studies before later external validation.

## Why OLYMPUS Exists

Modern security research is fragmented. Malware detection, social-engineering defense, LLM safety, deception, and digital forensics are often evaluated in isolation, with no shared platform for coordinating offensive and defensive reasoning. OLYMPUS exists to provide that platform:

- a common orchestration layer for heterogeneous cyber modules,
- a unified attacker/defender experimentation surface,
- persistent synthetic evidence for regression tracking,
- a bridge from prototype modules to future cyber-range or dataset-backed evaluation.

## System Architecture

| Layer | Role |
| --- | --- |
| `olympus/core/` | orchestration, logging, configuration, knowledge base |
| `olympus/modules/` | offense, defense, intelligence, and forensics modules |
| `olympus/api/` | FastAPI control and execution surface |
| `olympus/dashboard/` | Streamlit inspection and presentation layer |
| `experiments/` | synthetic comparisons, ablations, and export pipelines |
| `results/` | JSON summaries, LaTeX tables, and evidence notes |
| `tests/` | module-level behavioral checks |

### Offense-oriented and adaptive research modules

| Module | Purpose |
| --- | --- |
| Module 1 | penetration-testing and scanning workflows |
| Module 3 | zero-day discovery scaffolding and exploit-search research hooks |
| Module 4 | threat-intelligence aggregation |
| Module 6 | TITAN co-evolutionary attacker/defender optimization |

### Defense, assurance, and analysis modules

| Module | Purpose |
| --- | --- |
| Module 2 | malware detection and scoring |
| Module 5 | deception and honeypot-style defense research |
| Module 7 | social-engineering / phishing-style detection |
| Module 8 | AI model integrity checks |
| Module 9 | LLM jailbreak and prompt-defense workflows |
| Module 10 | digital forensics and post-event analysis |

## Adversarial Simulation And TITAN

The strongest methodological identity of OLYMPUS is not “best IDS on a public dataset.” It is **simulation-first adaptive security research**.

TITAN models attacker and defender strategy populations under a shared scoring loop. In this repository release, that loop is evaluated through seeded synthetic objectives and component ablations. That is legitimate for this research class because adaptive attacker/defender systems often need:

- controlled, repeatable adversarial environments,
- internal calibration before expensive external validation,
- direct study of strategy interactions rather than only static classification scores.

What matters is making the evidence type explicit. OLYMPUS now does that in code, JSON metadata, folder READMEs, and the paper.

## Evaluation Methodology

The repository currently contains two main evidence paths:

1. `experiments/baseline_comparison.py`
   - synthetic calibration studies for Modules 2, 6, 7, and 9
   - schema-rich JSON export with evidence type, metric semantics, reference method, and non-claims
   - LaTeX tables derived from the same machine-readable source

2. `experiments/ablation_study.py`
   - seeded TITAN component ablation under a shared co-evolutionary scoring model
   - significance metadata against `TITAN-Full`
   - explicit separation between proxy fitness metrics and real-world performance claims

The canonical saved artifacts are:

- [`results/comparison_tables/comparison_results.json`](results/comparison_tables/comparison_results.json)
- [`results/ablation/ablation_results.json`](results/ablation/ablation_results.json)
- [`results/README.md`](results/README.md)
- [`results/comparison_tables/README.md`](results/comparison_tables/README.md)
- [`results/ablation/README.md`](results/ablation/README.md)

## Results And Evidence

### Synthetic comparison summary

All values below are **synthetic calibration results**, not public-benchmark claims.

| Study | Evidence type | OLYMPUS result | Reference | Interpretation |
| --- | --- | --- | --- | --- |
| Module 2 malware calibration | synthetic benchmark-style calibration | `F1 = 0.9752 ± 0.0045` | Random Forest baseline `0.9026 ± 0.0142` | strong internal malware-stack calibration under the seeded toy task |
| Module 7 social-engineering calibration | synthetic adversarial-email calibration | `F1 = 0.9639 ± 0.0048` | SpamAssassin baseline `0.8124 ± 0.0067` | strong internal separation in the phishing-style toy task |
| Module 9 LLM-defense calibration | synthetic adversarial prompt-defense calibration | `F1 = 0.9499 ± 0.0042` | Pattern-matching-only `0.8547 ± 0.0036` | strong internal prompt-defense calibration; null-defense anchor remains `0.0000` |
| Module 6 TITAN evolution | synthetic co-evolutionary simulation | `best_fitness = 0.8500 ± 0.0165` | Standard-GA `0.6799 ± 0.0217` | TITAN leads the current shared synthetic search model |

Across the current ten-seed synthetic protocol, these OLYMPUS-vs-reference comparisons are statistically significant in the saved summaries for the reported primary metrics. That significance should be interpreted as **internal comparative signal inside the synthetic harness**, not as proof of external operational superiority.

### TITAN ablation summary

The TITAN ablation export is useful because it shows which pieces matter inside the current co-evolutionary loop.

| Configuration | Mean attack fitness | Status vs `TITAN-Full` | Interpretation |
| --- | --- | --- | --- |
| `TITAN-Full` | `0.8533 ± 0.0158` | reference | full co-evolutionary configuration |
| `TITAN-NoCoEvolution` | `0.7379 ± 0.0145` | significant drop (`p = 0.0051`) | strongest evidence that coupled attacker/defender evolution matters |
| `TITAN-NoCrossover` | `0.7742 ± 0.0158` | significant drop (`p = 0.0051`) | crossover materially contributes in the current search loop |
| `TITAN-NoMutation` | `0.8576 ± 0.0157` | not significant (`p = 0.3329`) | mutation removal is not a decisive failure mode in the current toy setup |
| `TITAN-HighMutation` | `0.8546 ± 0.0194` | not significant (`p = 0.6465`) | aggressive mutation does not clearly outperform or degrade the current baseline |

This is exactly the kind of result that is valuable for a simulation-first cyber-AI platform: it helps separate architectural contribution from presentation hype.

## Explicit Evidence Status

| Category | Current status | How to interpret it |
| --- | --- | --- |
| Platform implementation | real | the codebase, orchestration layer, API, CLI, dashboard, and modules are present |
| Comparison tables | synthetic | useful for internal calibration and reproducibility, not public benchmark claims |
| TITAN ablations | synthetic co-evolutionary | useful for algorithm attribution under a shared proxy objective |
| Real malware / phishing / jailbreak benchmark evidence | not yet committed | needed for public empirical superiority claims |
| Live offensive validation | future work | requires strict authorization and controlled environments |
| Cyber-range validation | future work | needed to elevate the repo beyond simulation-first evidence |

## What OLYMPUS Is Not Claiming

OLYMPUS is not currently claiming:

- state-of-the-art results on real operational malware, phishing, or jailbreak benchmarks,
- public benchmark leadership over fielded commercial security products,
- real-world autonomous red-team/blue-team superiority,
- deployment-grade SOC performance without further external validation.

Those boundaries are deliberate. They make the repo stronger, not weaker, because the current claims are matched by the saved evidence.

## Repository Layout

```text
olympus-security/
├── olympus/
│   ├── api/
│   ├── core/
│   ├── dashboard/
│   ├── modules/
│   └── cli.py
├── experiments/
│   ├── baseline_comparison.py
│   ├── ablation_study.py
│   └── statistical_tests.py
├── paper/
│   └── olympus_paper.tex
├── results/
│   ├── comparison_tables/
│   ├── ablation/
│   └── README.md
├── tests/
└── setup.py
```

## Reproducibility

Install the repository:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the synthetic comparison exports:

```bash
python experiments/baseline_comparison.py
python experiments/ablation_study.py
```

Run the module checks:

```bash
pytest tests/test_module7.py tests/test_module8.py tests/test_module9.py -q
```

The experiment scripts now write:

- schema-versioned JSON exports,
- folder-level evidence notes,
- LaTeX tables derived from the same saved summaries.

## Limitations

- The current evidence is synthetic or simulation-first.
- Module-level results are intentionally controlled and seeded rather than externally benchmarked.
- TITAN fitness is a proxy objective inside a shared search model, not a field measurement.
- Several offense pathways are research scaffolding and must remain authorization-bound.

## Future Work

- attach real malware, phishing, and LLM-jailbreak benchmark campaigns where appropriate
- validate TITAN and related modules in controlled cyber ranges
- expand module-level ablations beyond the currently exported studies
- connect synthetic strategy search to richer environment simulators and agent-based red/blue exercises

## Citation

```bibtex
@misc{tsitlauri2026olympussecurity,
  author = {George David Tsitlauri},
  title  = {OLYMPUS-SECURITY: AI-native cyber offense/defense research platform},
  year   = {2026},
  url    = {https://github.com/gdtsitlauri/olympus-security}
}
```

## License

MIT License.
