# OLYMPUS

**Autonomous Security Intelligence Research Platform**

**Author:** George David Tsitlauri  
**Contact:** gdtsitlauri@gmail.com  
**Website:** gdtsitlauri.dev  
**GitHub:** github.com/gdtsitlauri  
**Year:** 2026

OLYMPUS is a large, modular cybersecurity research platform that integrates offensive, defensive, and analytical security components under one codebase. Its central systems contribution is a shared orchestration layer plus the **TITAN** co-evolutionary attack/defense optimization module.

## Evidence Status

| Item | Current status |
| --- | --- |
| Multi-module security codebase | Present |
| FastAPI / dashboard / CLI surfaces | Present |
| Synthetic benchmark harnesses for selected modules | Present |
| Real external benchmark campaigns committed in this repo | Not yet present |
| Safe public claim | implemented platform with synthetic calibration evidence |

The current repository contains real code and real benchmark-export logic, but the principal comparison artifacts are synthetic calibration studies. They are useful and worth keeping, but they should not be mistaken for external, real-world security benchmark leadership.

## Overview

OLYMPUS is organized around:

- a shared orchestrator
- a shared knowledge base
- an audit/logging layer
- ten security-oriented modules
- a TITAN co-evolutionary engine for attack/defense strategy search

### Implemented module families

| Module | Area |
| --- | --- |
| Module 1 | penetration testing / scanning |
| Module 2 | malware detection |
| Module 3 | zero-day discovery scaffolding |
| Module 4 | threat intelligence |
| Module 5 | deception / honeypots |
| Module 6 | TITAN co-evolution |
| Module 7 | social-engineering detection |
| Module 8 | AI model integrity |
| Module 9 | LLM defense |
| Module 10 | digital forensics |

## What Is Real Here

The repository is not an empty architecture document. It includes:

- package code under `olympus/`
- a FastAPI REST API
- a Streamlit dashboard
- a CLI
- baseline-comparison and ablation scripts
- committed result tables and JSON exports for synthetic internal studies

## What The Current Results Mean

The result files under `results/` are currently best understood as:

- internal comparative calibration artifacts
- export-pipeline validation artifacts
- synthetic scenario studies for selected modules

They are **not** yet:

- external benchmark wins on real malware corpora
- deployment-grade phishing or jailbreak detection studies
- evidence of end-to-end operational security automation in the wild

See [results/README.md](results/README.md) for the evidence note.

## Repository Layout

```text
olympus-security/
├── olympus/
│   ├── core/
│   ├── modules/
│   ├── api/
│   ├── dashboard/
│   └── cli.py
├── experiments/
│   ├── baseline_comparison.py
│   └── ablation_study.py
├── paper/
│   └── olympus_paper.tex
├── results/
│   ├── comparison_tables/
│   └── ablation/
└── tests/
```

## Running The Repository

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Synthetic comparison / export run:

```bash
python experiments/baseline_comparison.py
```

The comparison script now writes metadata marking the exported files as
synthetic-calibration evidence.

## Real Vs Synthetic Vs Future Work

| Category | Status |
| --- | --- |
| Platform implementation | real and present |
| Module comparison tables | synthetic |
| Attack/defense co-evolution study | synthetic |
| Real malware / phishing / jailbreak benchmark campaigns | future work |
| Live offensive validation | future work and requires strict authorization |

## Limitations

- Current benchmark files are synthetic calibration artifacts.
- Several modules need real benchmark campaigns before strong empirical claims are justified.
- Offensive capabilities should be treated as research-only and authorization-bound.

## Citation

```bibtex
@misc{tsitlauri2026olympus,
  author = {George David Tsitlauri},
  title  = {OLYMPUS: Autonomous Security Intelligence Research Platform},
  year   = {2026},
  email  = {gdtsitlauri@gmail.com}
}
```

## License

MIT License.
