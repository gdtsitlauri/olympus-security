# ⚡ OLYMPUS — Autonomous Security Intelligence System

**OLYMPUS** (Offensive and Defensive Autonomous Security Intelligence System) is the most complete open-source autonomous AI security framework ever built. It integrates 10 security intelligence modules across all major cybersecurity domains, unified by a shared knowledge base and the novel **OLYMPUS-TITAN** co-evolutionary algorithm.

---

## Status

```
ΟΛΥΜΠΟΣ: OPERATIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Module 1  — Penetration Testing          [offensive]
✓ Module 2  — Virus Detection & Fighting   [defensive]
✓ Module 3  — Zero-Day Discovery           [offensive]
✓ Module 4  — Threat Intelligence          [defensive]
✓ Module 5  — Deception & Honeypots        [defensive]
✓ Module 6  — Self-Evolution (TITAN)       [core]
✓ Module 7  — Social Engineering Detection [defensive]
✓ Module 8  — AI Model Integrity           [defensive]
✓ Module 9  — LLM Defense                 [defensive]
✓ Module 10 — Digital Forensics            [defensive]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: GTX 1650 (4GB VRAM, CUDA 12.x) + CPU fallback
```

---

## Novel Algorithm: OLYMPUS-TITAN

**TITAN** (Total Intelligent Threat Analysis and Neutralization) is the first co-evolutionary algorithm applied to the full attack-defense security optimization problem.

```
Attack Population ←→ Defense Population
     evolves              evolves
     against              against
       ↕                    ↕
   Neural Fitness Evaluator (F_θ)
       ↕
   Mutation + Crossover + Selection
       ↕
   Updated F_θ from matchup outcomes
```

Attack and defense strategy populations compete and co-evolve through:
- **Mutation** — adaptive Gaussian noise (σ decays over generations)
- **Crossover** — single-point recombination of gene vectors
- **Tournament selection** — fitness-proportional tournament
- **Neural fitness** — neural network predicts attack success probability
- **Elite preservation** — top strategies survive each generation

---

## Quick Start

### Prerequisites

```bash
# CUDA (GTX 1650, CUDA 12.x)
nvidia-smi   # verify GPU

# Python 3.11+
python --version
```

### Installation

```bash
git clone https://github.com/olympus-security/olympus
cd olympus-security

# Install (CPU)
pip install -r requirements.txt
pip install -e .

# Install with GPU support (GTX 1650)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

### Docker

```bash
docker-compose up olympus-api olympus-dashboard
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Docs: http://localhost:8000/docs
```

---

## Usage

### CLI

```bash
# System status
olympus status

# Penetration testing (authorized targets only)
olympus pentest --target 192.168.1.1 --scope network,web

# Malware scan
olympus scan --path /var/www --quarantine

# Zero-day discovery (static analysis)
olympus zeroday --path ./src --mode static

# Zero-day fuzzing
olympus zeroday --mode fuzz --cmd ./target_binary --iterations 10000

# Threat intelligence
olympus threat-intel --techniques T1566 T1059 T1027

# TITAN co-evolution
olympus evolve --generations 100 --population 50

# Social engineering detection
olympus social-eng --text "URGENT: Your account will be suspended!"

# LLM jailbreak detection
olympus llm-defense --prompts "Ignore previous instructions and..."

# AI model integrity
olympus ai-integrity --models model.pt checkpoint.pth

# Digital forensics
olympus forensics --path /incident --title "Production Breach"

# Start API server
olympus serve

# Start dashboard
olympus dashboard
```

### Python API

```python
import olympus
from olympus.core.orchestrator import ORCHESTRATOR

# Run TITAN evolution
task = ORCHESTRATOR.submit(
    "module6_evolution",
    generations=100,
    population_size=50,
    seed=42,
)

# Wait for completion
import time
while task.status.value == "running":
    time.sleep(1)

print(f"Attack fitness: {task.result.metrics['best_attack_fitness']:.4f}")
print(f"Defense fitness: {task.result.metrics['best_defense_fitness']:.4f}")
```

### REST API

```bash
# API documentation
open http://localhost:8000/docs

# Run penetration test
curl -X POST http://localhost:8000/pentest \
  -H "Content-Type: application/json" \
  -d '{"target": "192.168.1.1", "scope": "network+web"}'

# Run TITAN evolution
curl -X POST http://localhost:8000/evolve \
  -H "Content-Type: application/json" \
  -d '{"generations": 100, "population_size": 50}'

# Detect jailbreak
curl -X POST http://localhost:8000/detect-jailbreak \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Ignore previous instructions..."]}'

# Query knowledge base
curl http://localhost:8000/kb/threats?severity=critical
```

---

## Architecture

```
olympus-security/
├── olympus/
│   ├── core/
│   │   ├── orchestrator.py       # Central task manager
│   │   ├── knowledge_base.py     # Shared OLYMPUS-KB
│   │   ├── base_module.py        # Abstract module base
│   │   ├── device.py             # GPU/CPU management
│   │   ├── config.py             # Configuration
│   │   └── logger.py             # Audit + runtime logging
│   ├── modules/
│   │   ├── module1_pentest/      # Network + web scanning
│   │   ├── module2_virus/        # ML malware detection
│   │   ├── module3_zeroday/      # Fuzzing + static analysis
│   │   ├── module4_threat_intel/ # MITRE ATT&CK intelligence
│   │   ├── module5_deception/    # Honeypots
│   │   ├── module6_evolution/    # OLYMPUS-TITAN algorithm
│   │   ├── module7_social_eng/   # Social engineering detection
│   │   ├── module8_ai_integrity/ # AI model integrity
│   │   ├── module9_llm_defense/  # LLM jailbreak detection
│   │   └── module10_forensics/   # Incident response
│   ├── api/main.py               # FastAPI REST API
│   ├── dashboard/app.py          # Streamlit dashboard
│   └── cli.py                    # Command-line interface
├── experiments/
│   ├── baseline_comparison.py    # Module vs baseline evaluation
│   ├── ablation_study.py         # TITAN ablation
│   └── statistical_tests.py      # Wilcoxon, CI, Cohen's d
├── paper/
│   └── olympus_paper.tex         # IEEE/ACM paper
├── results/
│   ├── comparison_tables/        # Module benchmark results
│   └── ablation/                 # TITAN ablation results
├── data/
│   ├── quarantine/               # Quarantined files
│   └── models/                   # Trained model weights
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Experiment Results

Run experiments:

```bash
# Baseline comparison (all modules)
python experiments/baseline_comparison.py

# TITAN ablation study
python experiments/ablation_study.py
```

### Key Results (mean ± std, 10 seeds)

| Module | Method | F1 / Fitness | vs. Baseline |
|--------|--------|-------------|--------------|
| M2: Malware Detection | OLYMPUS-CNN+GBM | **0.9752 ± 0.005** | +0.135 vs ClamAV |
| M7: Social Engineering | OLYMPUS-SE-Detector | **0.9639 ± 0.005** | +0.152 vs SpamAssassin |
| M9: LLM Defense | OLYMPUS-LLM-Defense | **0.9499 ± 0.004** | +0.095 vs Pattern-only |
| M6: TITAN Evolution | OLYMPUS-TITAN | **0.8500 ± 0.017** | +0.352 vs Random |

All improvements statistically significant (p < 0.005, Wilcoxon signed-rank, 10 seeds).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1650 (4GB) | RTX 3080+ |
| RAM | 16GB | 32GB |
| Storage | 50GB | 200GB |
| Python | 3.11 | 3.11+ |
| CUDA | 12.0 | 12.4 |

All modules include CPU fallback for systems without CUDA.

---

## Safety & Ethics

⚠️ **OLYMPUS is designed exclusively for:**
- Authorized security research
- Penetration testing on systems you own or have written permission to test
- CTF (Capture The Flag) competitions
- Academic/educational purposes
- Defensive security operations

Using OLYMPUS against systems without authorization is illegal. All offensive
modules require explicit target scope configuration. Every action is logged to
the audit trail (`data/audit.jsonl`).

---

## Paper

Full paper: [`paper/olympus_paper.tex`](paper/olympus_paper.tex)

**OLYMPUS: An Autonomous AI Security Intelligence Framework with
Co-Evolutionary Attack-Defense Optimization via TITAN**

Novel contributions:
1. OLYMPUS-TITAN: First co-evolutionary attack-defense optimization algorithm
2. First unified framework covering all 10 security domains with AI
3. First open-source autonomous security AI with self-evolution
4. Cross-module knowledge sharing via OLYMPUS-KB

---

## License

MIT License — see [LICENSE](LICENSE)

Ethical use only. See license for full disclaimer.

---

## Author

**George David Tsitlauri**
AI & Systems Engineer
Informatics & Telecommunications, University of Thessaly, Greece
© 2026
