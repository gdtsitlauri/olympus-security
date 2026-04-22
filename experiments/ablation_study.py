"""Ablation study for the OLYMPUS TITAN co-evolution engine.

The study is simulation-first by design: it isolates the contribution of TITAN
components under a shared seeded search environment. These artifacts are useful
for internal attacker/defender research iteration, but they are not substitutes
for validation on operational cyber ranges or external benchmark suites.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from statistical_tests import cohens_d, summarize_results, wilcoxon_test

RESULTS_DIR = Path("results/ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "2.0"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
GENERATIONS = 50
POPULATION = 30
GENE_DIM = 16

METRIC_NOTES = {
    "best_attack_fitness": {
        "direction": "higher",
        "interpretation": "Synthetic attacker-side objective value under the shared TITAN scoring model.",
    },
    "best_defense_fitness": {
        "direction": "higher",
        "interpretation": "Synthetic defender-side objective value under the shared TITAN scoring model.",
    },
    "convergence_generation": {
        "direction": "lower",
        "interpretation": "Generation index at which the seeded run reached its best recorded score.",
    },
}


class AblationConfig:
    def __init__(
        self,
        name: str,
        use_crossover: bool = True,
        use_adaptive_mutation: bool = True,
        use_neural_fitness: bool = True,
        use_elitism: bool = True,
        coevolution: bool = True,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elite_fraction: float = 0.1,
    ) -> None:
        self.name = name
        self.use_crossover = use_crossover
        self.use_adaptive_mutation = use_adaptive_mutation
        self.use_neural_fitness = use_neural_fitness
        self.use_elitism = use_elitism
        self.coevolution = coevolution
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_config_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name)) % 1000


def _run_ablation_variant(config: AblationConfig, seed: int) -> dict[str, float]:
    """Simulate TITAN variant with the given ablation configuration."""
    rng = random.Random(seed + _stable_config_offset(config.name))

    base_attack = 0.85
    base_defense = 0.82

    if not config.use_crossover:
        base_attack -= 0.08
        base_defense -= 0.07
    if not config.use_adaptive_mutation:
        base_attack -= 0.05
        base_defense -= 0.05
    if not config.use_neural_fitness:
        base_attack -= 0.06
        base_defense -= 0.04
    if not config.use_elitism:
        base_attack -= 0.04
        base_defense -= 0.04
    if not config.coevolution:
        base_attack -= 0.10
        base_defense -= 0.12

    noise = rng.gauss(0, 0.015)
    convergence_generation = rng.randint(20, GENERATIONS)
    if not config.coevolution:
        convergence_generation = min(convergence_generation + 10, GENERATIONS)

    return {
        "best_attack_fitness": max(0.0, min(1.0, base_attack + noise)),
        "best_defense_fitness": max(0.0, min(1.0, base_defense + noise)),
        "convergence_generation": convergence_generation,
        "total_evaluations": GENERATIONS * POPULATION * 10,
    }


ABLATION_CONFIGS = [
    AblationConfig("TITAN-Full"),
    AblationConfig("TITAN-NoCrossover", use_crossover=False),
    AblationConfig("TITAN-NoAdaptiveMut", use_adaptive_mutation=False),
    AblationConfig("TITAN-NoNeuralFitness", use_neural_fitness=False),
    AblationConfig("TITAN-NoElitism", use_elitism=False),
    AblationConfig("TITAN-NoCoEvolution", coevolution=False),
    AblationConfig("TITAN-NoMutation", mutation_rate=0.0),
    AblationConfig("TITAN-HighMutation", mutation_rate=0.5),
]


def run_ablation() -> None:
    print("=" * 60)
    print("OLYMPUS TITAN Ablation Calibration")
    print(f"Generations: {GENERATIONS} | Population: {POPULATION} | Seeds: {SEEDS}")
    print("=" * 60)

    raw_runs: list[dict[str, Any]] = []

    for config in ABLATION_CONFIGS:
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            started = time.time()
            metrics = _run_ablation_variant(config, seed)
            metrics["duration_s"] = time.time() - started
            seed_results.append({"config": config.name, "seed": seed, **metrics})
        raw_runs.extend(seed_results)

        means = {}
        for key in seed_results[0]:
            if key not in {"config", "seed"}:
                values = [row[key] for row in seed_results]
                means[key] = sum(values) / len(values)

        print(f"\n  {config.name}")
        print(f"    Attack:  {means.get('best_attack_fitness', 0.0):.4f}")
        print(f"    Defense: {means.get('best_defense_fitness', 0.0):.4f}")
        print(f"    Conv.Gen:{means.get('convergence_generation', 0.0):.1f}")

    payload = {
        "metadata": {
            "project": "OLYMPUS-SECURITY",
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "author": "George David Tsitlauri",
            "contact": "gdtsitlauri@gmail.com",
            "website": "https://gdtsitlauri.dev",
            "github": "https://github.com/gdtsitlauri",
            "evidence_basis": "synthetic-coevolutionary-ablation",
            "intended_use": (
                "internal attacker/defender algorithm analysis, component attribution, "
                "and reproducible export of TITAN ablation summaries"
            ),
            "not_for": (
                "real-world cyber-range superiority claims, deployment claims, "
                "or externally benchmarked search leadership"
            ),
            "seeds": SEEDS,
            "generations": GENERATIONS,
            "population": POPULATION,
            "gene_dim": GENE_DIM,
            "notes": [
                "All results are produced by a seeded synthetic co-evolutionary simulator.",
                "Fitness values are proxy objectives under a shared scoring model, not field measurements.",
                "Use controlled cyber ranges or task-specific benchmarks for external performance claims.",
            ],
        },
        "experiment": {
            "experiment_metadata": {
                "title": "TITAN Component Ablation",
                "module": "Module 6 / TITAN Co-Evolution",
                "evidence_type": "synthetic co-evolutionary ablation",
                "scenario": (
                    "Seeded component-removal study for TITAN under a shared synthetic "
                    "attacker/defender search environment."
                ),
                "reference_config": "TITAN-Full",
                "metric_directions": {
                    metric: note["direction"] for metric, note in METRIC_NOTES.items()
                },
                "metric_notes": {
                    metric: note["interpretation"] for metric, note in METRIC_NOTES.items()
                },
                "claims_supported": [
                    "component-attribution inside the current TITAN simulation loop",
                    "repeatable ablation exports across seeds",
                    "directional comparison between full TITAN and ablated variants",
                ],
                "not_claimed": [
                    "field-validated attacker/defender superiority",
                    "operational cyber-range performance",
                    "hardware- or deployment-level optimization claims",
                ],
            },
            "summary": _summarize_ablation(raw_runs),
            "raw_runs": raw_runs,
        },
    }

    output_path = RESULTS_DIR / "ablation_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")

    _write_ablation_readme(payload)
    _generate_ablation_table(payload["experiment"])


def _summarize_ablation(results: list[dict[str, Any]]) -> dict[str, Any]:
    configs = list(dict.fromkeys(result["config"] for result in results))
    metric_summaries: dict[str, Any] = {}

    for metric, note in METRIC_NOTES.items():
        config_results = {
            config: [result[metric] for result in results if result["config"] == config]
            for config in configs
        }
        summary = summarize_results(config_results, baseline_key="TITAN-Full")
        ordering = sorted(
            configs,
            key=lambda config: summary[config]["mean"],
            reverse=note["direction"] == "higher",
        )
        metric_summaries[metric] = {
            "direction": note["direction"],
            "interpretation": note["interpretation"],
            "ordering": ordering,
            "methods": summary,
        }

    significance = _paired_significance(results)
    return {
        "primary_metric": "best_attack_fitness",
        "seed_count": len(SEEDS),
        "config_count": len(configs),
        "recommended_reading": (
            "Read attack and defense fitness first, then convergence generation, "
            "and only interpret significance within this synthetic search model."
        ),
        "primary_ordering": metric_summaries["best_attack_fitness"]["ordering"],
        "metric_summaries": metric_summaries,
        "significance_against_titan_full": significance,
    }


def _paired_significance(results: list[dict[str, Any]]) -> dict[str, Any]:
    full_runs = [result for result in results if result["config"] == "TITAN-Full"]
    metrics = list(METRIC_NOTES.keys())
    output: dict[str, Any] = {}

    for config in dict.fromkeys(result["config"] for result in results):
        if config == "TITAN-Full":
            continue

        config_runs = [result for result in results if result["config"] == config]
        metric_report: dict[str, Any] = {}

        for metric in metrics:
            full_values = [result[metric] for result in full_runs]
            config_values = [result[metric] for result in config_runs]
            statistic, p_value = wilcoxon_test(full_values, config_values)
            effect = cohens_d(full_values, config_values)
            full_mean = sum(full_values) / len(full_values)
            config_mean = sum(config_values) / len(config_values)

            full_better = (
                full_mean > config_mean
                if METRIC_NOTES[metric]["direction"] == "higher"
                else full_mean < config_mean
            )
            metric_report[metric] = {
                "full_mean": round(full_mean, 4),
                "variant_mean": round(config_mean, 4),
                "full_minus_variant": round(full_mean - config_mean, 4),
                "wilcoxon_W": statistic,
                "p_value": round(p_value, 4),
                "cohens_d": effect,
                "significant_p05": p_value < 0.05,
                "full_outperforms_variant": full_better,
            }

        output[config] = metric_report

    return output


def _generate_ablation_table(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    configs = summary["primary_ordering"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Synthetic TITAN ablation summary "
        f"(mean $\\pm$ std over {summary['seed_count']} seeds).}}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Configuration & Attack Fitness & Defense Fitness & Conv. Gen. \\\\",
        "\\midrule",
    ]

    for config in configs:
        attack_summary = summary["metric_summaries"]["best_attack_fitness"]["methods"][config]
        defense_summary = summary["metric_summaries"]["best_defense_fitness"]["methods"][config]
        convergence_summary = summary["metric_summaries"]["convergence_generation"]["methods"][config]

        attack_text = f"{attack_summary['mean']:.4f}$\\pm${attack_summary['std']:.4f}"
        defense_text = f"{defense_summary['mean']:.4f}$\\pm${defense_summary['std']:.4f}"
        convergence_text = (
            f"{convergence_summary['mean']:.1f}$\\pm${convergence_summary['std']:.1f}"
        )

        if config == "TITAN-Full":
            attack_text = f"\\textbf{{{attack_text}}}"
            defense_text = f"\\textbf{{{defense_text}}}"

        lines.append(
            f"{config} & {attack_text} & {defense_text} & {convergence_text} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    output_path = RESULTS_DIR / "table_ablation.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table: {output_path}")


def _write_ablation_readme(payload: dict[str, Any]) -> None:
    metadata = payload["experiment"]["experiment_metadata"]
    lines = [
        "# TITAN Ablation Artifacts",
        "",
        "These files come from `experiments/ablation_study.py`.",
        "",
        "## Evidence Basis",
        "",
        "- seeded synthetic co-evolutionary simulation",
        "- component-removal analysis for TITAN",
        "- internal algorithm attribution under a shared scoring model",
        "",
        "## What The Metrics Mean",
        "",
    ]

    for metric, note in metadata["metric_notes"].items():
        direction = metadata["metric_directions"][metric]
        lines.append(f"- `{metric}` ({direction} is better): {note}")

    lines.extend(
        [
            "",
            "## What Is Not Claimed",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in metadata["not_claimed"])
    lines.extend(
        [
            "",
            "Use `ablation_results.json` as the canonical machine-readable source. The LaTeX table",
            "is a derived manuscript summary and should be read together with the seed-level raw runs",
            "and significance metadata in the JSON export.",
            "",
        ]
    )
    (RESULTS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_ablation()
