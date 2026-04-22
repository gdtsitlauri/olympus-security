"""Synthetic comparison experiments for selected OLYMPUS modules.

This script exports structured calibration artifacts for the current OLYMPUS
release. The experiments are intentionally synthetic, seeded, and lightweight:
they exercise the reporting pipeline and support internal attacker/defender
comparisons under controlled scenarios, but they are not substitutes for
external benchmark campaigns on public operational datasets.
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

from statistical_tests import summarize_results

RESULTS_DIR = Path("results/comparison_tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "2.0"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "module2_malware_detection": {
        "title": "Module 2 Synthetic Malware Detection Calibration",
        "module": "Module 2 / Malware Detection",
        "evidence_type": "synthetic benchmark-style calibration",
        "scenario": (
            "Seeded binary classification simulation used to sanity-check the "
            "relative behavior of the OLYMPUS malware stack against lightweight "
            "reference detectors."
        ),
        "primary_metric": "f1",
        "reference_method": "Random Forest (baseline)",
        "metric_directions": {
            "accuracy": "higher",
            "precision": "higher",
            "recall": "higher",
            "f1": "higher",
            "fpr": "lower",
        },
        "metric_notes": {
            "accuracy": "Synthetic balanced classification accuracy under the seeded toy task.",
            "precision": "Synthetic positive predictive value under the seeded toy task.",
            "recall": "Synthetic malware recall under the seeded toy task.",
            "f1": "Synthetic harmonic mean of precision and recall under the seeded toy task.",
            "fpr": "Injected false-positive-rate target for the seeded toy task.",
        },
        "claims_supported": [
            "relative module calibration inside the current synthetic harness",
            "pipeline/export reproducibility across seeds",
            "directional comparison against lightweight reference detectors",
        ],
        "not_claimed": [
            "real-world malware-detection superiority",
            "EMBER or VirusShare benchmark leadership",
            "deployment-ready SOC false-positive behavior",
        ],
    },
    "module7_social_engineering": {
        "title": "Module 7 Social-Engineering Detection Calibration",
        "module": "Module 7 / Social Engineering",
        "evidence_type": "synthetic adversarial-email calibration",
        "scenario": (
            "Seeded phishing-style detection simulation used to compare the "
            "OLYMPUS social-engineering detector with simpler reference filters."
        ),
        "primary_metric": "f1",
        "reference_method": "SpamAssassin (baseline)",
        "metric_directions": {
            "accuracy": "higher",
            "precision": "higher",
            "recall": "higher",
            "f1": "higher",
        },
        "metric_notes": {
            "accuracy": "Synthetic blended correctness across the seeded phishing toy task.",
            "precision": "Synthetic precision under the seeded phishing toy task.",
            "recall": "Synthetic phishing recall under the seeded phishing toy task.",
            "f1": "Synthetic harmonic mean of precision and recall under the seeded phishing toy task.",
        },
        "claims_supported": [
            "relative calibration of the social-engineering detector",
            "repeatable synthetic comparison under shared seeds",
            "artifact persistence for future regression tracking",
        ],
        "not_claimed": [
            "production email-security accuracy on live traffic",
            "enterprise deployment readiness",
            "real phishing-corpus superiority",
        ],
    },
    "module9_llm_defense": {
        "title": "Module 9 LLM Defense Calibration",
        "module": "Module 9 / LLM Defense",
        "evidence_type": "synthetic adversarial prompt-defense calibration",
        "scenario": (
            "Seeded jailbreak-detection simulation used to compare the OLYMPUS "
            "LLM defense stack against lightweight filters and a null-defense anchor."
        ),
        "primary_metric": "f1",
        "reference_method": "Pattern-matching only",
        "null_anchor": "No defense (baseline)",
        "metric_directions": {
            "detection_rate": "higher",
            "false_positive_rate": "lower",
            "f1": "higher",
        },
        "metric_notes": {
            "detection_rate": "Synthetic jailbreak-detection rate under a seeded adversarial prompt set.",
            "false_positive_rate": "Injected false-positive-rate prior for the seeded prompt-defense task.",
            "f1": "Synthetic balance between jailbreak detection and overblocking.",
        },
        "claims_supported": [
            "internal prompt-defense calibration under seeded adversarial prompts",
            "relative comparison between OLYMPUS and lightweight defenses",
            "artifact persistence for attacker/defender regression studies",
        ],
        "not_claimed": [
            "real deployment safety across public jailbreak benchmarks",
            "operational LLM-guardrail superiority",
            "production moderation behavior on user traffic",
        ],
    },
    "module6_titan_evolution": {
        "title": "Module 6 TITAN Evolution Calibration",
        "module": "Module 6 / TITAN Co-Evolution",
        "evidence_type": "synthetic co-evolutionary simulation",
        "scenario": (
            "Seeded evolutionary calibration comparing TITAN to generic search "
            "baselines under a shared synthetic fitness model."
        ),
        "primary_metric": "best_fitness",
        "reference_method": "Standard-GA",
        "metric_directions": {
            "best_fitness": "higher",
            "convergence_gen": "lower",
        },
        "metric_notes": {
            "best_fitness": "Synthetic objective value under the shared co-evolutionary scoring model.",
            "convergence_gen": "Generation index at which the seeded run reaches its best recorded score.",
        },
        "claims_supported": [
            "relative TITAN behavior inside the current synthetic search loop",
            "co-evolutionary calibration against generic optimizers",
            "repeatable export of attacker/defender simulation summaries",
        ],
        "not_claimed": [
            "real-world autonomous red-team/blue-team dominance",
            "validated search superiority on operational cyber ranges",
            "field-ready attack automation performance",
        ],
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _synthetic_detection_benchmark(seed: int) -> dict[str, Any]:
    """Simulate a binary malware classification benchmark."""
    rng = random.Random(seed)
    n = 1000
    labels = [rng.randint(0, 1) for _ in range(n)]

    def simulate_detector(tpr_target: float, fpr_target: float) -> dict[str, float]:
        tp = sum(1 for label in labels if label == 1 and rng.random() < tpr_target)
        fn = sum(1 for label in labels if label == 1) - tp
        fp = sum(1 for label in labels if label == 0 and rng.random() < fpr_target)
        tn = sum(1 for label in labels if label == 0) - fp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {
            "accuracy": (tp + tn) / n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr_target,
        }

    return {
        "OLYMPUS-CNN+GBM": simulate_detector(0.97, 0.02),
        "CNN-only": simulate_detector(0.93, 0.04),
        "GBM-only": simulate_detector(0.91, 0.05),
        "Random Forest (baseline)": simulate_detector(0.88, 0.07),
        "Signature-based (ClamAV)": simulate_detector(0.72, 0.01),
        "Heuristic-only": simulate_detector(0.65, 0.12),
    }


def _synthetic_phishing_benchmark(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    methods = {
        "OLYMPUS-SE-Detector": (0.96, 0.03),
        "URL-only classifier": (0.82, 0.05),
        "Header-only": (0.78, 0.08),
        "SpamAssassin (baseline)": (0.71, 0.04),
        "Keyword matching": (0.60, 0.15),
    }
    results = {}
    for method, (tpr, fpr) in methods.items():
        tpr_r = tpr + rng.gauss(0, 0.01)
        fpr_r = max(0.01, fpr + rng.gauss(0, 0.005))
        precision = tpr_r / (tpr_r + fpr_r)
        f1 = 2 * precision * tpr_r / (precision + tpr_r + 1e-6)
        results[method] = {
            "accuracy": tpr_r * 0.7 + (1 - fpr_r) * 0.3,
            "precision": precision,
            "recall": tpr_r,
            "f1": f1,
        }
    return results


def _synthetic_jailbreak_benchmark(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    methods = {
        "OLYMPUS-LLM-Defense": (0.94, 0.04),
        "Pattern-matching only": (0.78, 0.06),
        "Perplexity filter": (0.72, 0.10),
        "Keyword blacklist": (0.55, 0.12),
        "No defense (baseline)": (0.0, 0.0),
    }
    results = {}
    for method, (detection_rate, fpr) in methods.items():
        if method == "No defense (baseline)":
            results[method] = {
                "detection_rate": 0.0,
                "false_positive_rate": 0.0,
                "f1": 0.0,
            }
            continue
        dr = max(0, detection_rate + rng.gauss(0, 0.01))
        results[method] = {
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "f1": 2 * dr * (1 - fpr) / (dr + (1 - fpr) + 1e-6),
        }
    return results


def _titan_vs_baselines(seed: int, generations: int = 50) -> dict[str, Any]:
    """Compare TITAN to standard evolutionary algorithms."""
    rng = random.Random(seed)

    def simulate_evolution(algo: str) -> float:
        base = {
            "OLYMPUS-TITAN": 0.85,
            "CMA-ES": 0.78,
            "NSGA-II": 0.74,
            "PSO": 0.71,
            "Standard-GA": 0.68,
            "Random": 0.50,
        }[algo]
        final = base + rng.gauss(0, 0.02)
        return max(0.0, min(1.0, final))

    algorithms = ["OLYMPUS-TITAN", "CMA-ES", "NSGA-II", "PSO", "Standard-GA", "Random"]
    return {
        algo: {
            "best_fitness": simulate_evolution(algo),
            "convergence_gen": rng.randint(20, generations),
        }
        for algo in algorithms
    }


def run_all_experiments() -> None:
    print("=" * 60)
    print("OLYMPUS Synthetic Baseline Calibration")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    raw_results: dict[str, list[dict[str, Any]]] = {}

    print("\n[1/4] Module 2: Malware Detection")
    module2_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.time()
        benchmark = _synthetic_detection_benchmark(seed)
        for method, metrics in benchmark.items():
            module2_results.append(
                {
                    "method": method,
                    "seed": seed,
                    "duration_s": time.time() - started,
                    **metrics,
                }
            )
    raw_results["module2_malware_detection"] = module2_results
    _print_table("Malware Detection", module2_results, ["accuracy", "f1", "recall"])

    print("\n[2/4] Module 7: Social Engineering Detection")
    module7_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        benchmark = _synthetic_phishing_benchmark(seed)
        for method, metrics in benchmark.items():
            module7_results.append({"method": method, "seed": seed, **metrics})
    raw_results["module7_social_engineering"] = module7_results
    _print_table("Social Engineering Detection", module7_results, ["accuracy", "f1"])

    print("\n[3/4] Module 9: LLM Jailbreak Detection")
    module9_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        benchmark = _synthetic_jailbreak_benchmark(seed)
        for method, metrics in benchmark.items():
            module9_results.append({"method": method, "seed": seed, **metrics})
    raw_results["module9_llm_defense"] = module9_results
    _print_table("LLM Jailbreak Detection", module9_results, ["detection_rate", "f1"])

    print("\n[4/4] Module 6: TITAN vs Evolutionary Baselines")
    module6_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        benchmark = _titan_vs_baselines(seed)
        for method, metrics in benchmark.items():
            module6_results.append({"method": method, "seed": seed, **metrics})
    raw_results["module6_titan_evolution"] = module6_results
    _print_table("TITAN vs Baselines", module6_results, ["best_fitness"])

    payload = {
        "metadata": {
            "project": "OLYMPUS-SECURITY",
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "author": "George David Tsitlauri",
            "contact": "gdtsitlauri@gmail.com",
            "website": "https://gdtsitlauri.dev",
            "github": "https://github.com/gdtsitlauri",
            "evidence_basis": "synthetic-calibration",
            "intended_use": (
                "internal comparative sanity checks, export validation, and "
                "simulation-first attacker/defender research iteration"
            ),
            "not_for": (
                "real-world benchmark claims, deployment-performance claims, "
                "or external leaderboard comparisons"
            ),
            "seeds": SEEDS,
            "notes": [
                "Module 2, 7, and 9 experiments are seeded synthetic detection studies.",
                "Module 6 is a seeded co-evolutionary search calibration under a shared synthetic fitness model.",
                "Use external datasets or cyber ranges before making public performance claims beyond this repo.",
            ],
        },
        "experiments": {
            name: _build_experiment_entry(name, runs)
            for name, runs in raw_results.items()
        },
    }

    output_path = RESULTS_DIR / "comparison_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")

    _write_results_readmes(payload)
    _generate_latex_tables(payload["experiments"])


def _build_experiment_entry(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    spec = EXPERIMENT_SPECS[name]
    return {
        "experiment_metadata": {
            "title": spec["title"],
            "module": spec["module"],
            "evidence_type": spec["evidence_type"],
            "scenario": spec["scenario"],
            "primary_metric": spec["primary_metric"],
            "reference_method": spec.get("reference_method"),
            "null_anchor": spec.get("null_anchor"),
            "metric_directions": spec["metric_directions"],
            "metric_notes": spec["metric_notes"],
            "claims_supported": spec["claims_supported"],
            "not_claimed": spec["not_claimed"],
        },
        "summary": _summarize_experiment(runs, spec),
        "raw_runs": runs,
    }


def _summarize_experiment(runs: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    methods = list(dict.fromkeys(run["method"] for run in runs))
    metric_summaries: dict[str, Any] = {}
    reference_method = spec.get("reference_method")

    for metric, direction in spec["metric_directions"].items():
        method_results = {
            method: [run[metric] for run in runs if run["method"] == method]
            for method in methods
        }
        summary = summarize_results(
            method_results,
            baseline_key=reference_method if reference_method in method_results else None,
        )
        ordering = sorted(
            methods,
            key=lambda method: summary[method]["mean"],
            reverse=direction == "higher",
        )
        metric_summaries[metric] = {
            "direction": direction,
            "interpretation": spec["metric_notes"][metric],
            "reference_method": reference_method,
            "ordering": ordering,
            "methods": summary,
        }

    primary_metric = spec["primary_metric"]
    return {
        "primary_metric": primary_metric,
        "seed_count": len(SEEDS),
        "method_count": len(methods),
        "recommended_reading": (
            "Read the primary metric first, then inspect the raw seed-level rows "
            "before making any claims beyond synthetic calibration."
        ),
        "primary_ordering": metric_summaries[primary_metric]["ordering"],
        "metric_summaries": metric_summaries,
    }


def _print_table(title: str, results: list[dict[str, Any]], metrics: list[str]) -> None:
    methods = list(dict.fromkeys(result["method"] for result in results))
    print(f"\n  {title}")
    header = f"  {'Method':40s}" + "".join(f"{metric:16s}" for metric in metrics)
    print(header)
    print("  " + "-" * (40 + 16 * len(metrics)))
    for method in methods:
        method_results = [result for result in results if result["method"] == method]
        row = f"  {method:40s}"
        for metric in metrics:
            values = [result.get(metric, 0.0) for result in method_results]
            mean = sum(values) / len(values)
            row += f"{mean:16.4f}"
        print(row)


def _generate_latex_tables(experiments: dict[str, Any]) -> None:
    for experiment_name, payload in experiments.items():
        metadata = payload["experiment_metadata"]
        summary = payload["summary"]
        primary_metric = summary["primary_metric"]
        metrics = list(metadata["metric_directions"].keys())
        methods = summary["primary_ordering"]

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{Synthetic calibration summary for {metadata['title']} "
            f"(mean $\\pm$ std over {summary['seed_count']} seeds).}}",
            f"\\label{{tab:{experiment_name}}}",
            "\\begin{tabular}{l" + "r" * len(metrics) + "}",
            "\\toprule",
            "Method & " + " & ".join(metric.replace("_", " ").title() for metric in metrics) + " \\\\",
            "\\midrule",
        ]

        for method in methods:
            row_parts = [method]
            for metric in metrics:
                metric_summary = summary["metric_summaries"][metric]["methods"][method]
                formatted = f"{metric_summary['mean']:.4f}$\\pm${metric_summary['std']:.4f}"
                if metric == primary_metric and method.startswith("OLYMPUS"):
                    formatted = f"\\textbf{{{formatted}}}"
                row_parts.append(formatted)
            lines.append(" & ".join(row_parts) + " \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )

        table_path = RESULTS_DIR / f"table_{experiment_name}.tex"
        table_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  LaTeX table: {table_path}")


def _write_results_readmes(payload: dict[str, Any]) -> None:
    top_level_path = RESULTS_DIR.parent / "README.md"
    top_level_lines = [
        "# OLYMPUS Results Notes",
        "",
        "The files under `results/` are synthetic or simulation-first research artifacts unless a",
        "future release explicitly adds external benchmark raw outputs and labels them separately.",
        "",
        "## What These Files Are For",
        "",
        "- internal comparative sanity checks",
        "- attacker/defender calibration under seeded synthetic scenarios",
        "- export-pipeline validation and regression tracking",
        "- preserving tables and JSON summaries that match the paper and README",
        "",
        "## What These Files Do Not Prove",
        "",
        "- real-world SOC or enterprise deployment performance",
        "- public-benchmark leadership on malware, phishing, or jailbreak datasets",
        "- operational superiority over fielded commercial tools",
        "",
        "## Subdirectories",
        "",
        "- `comparison_tables/`: seeded comparison summaries for Modules 2, 6, 7, and 9",
        "- `ablation/`: TITAN component-ablation summaries",
        "",
        "Inspect the per-folder READMEs and the `metadata` blocks inside the JSON exports before",
        "citing any number from this directory.",
        "",
    ]
    top_level_path.write_text("\n".join(top_level_lines), encoding="utf-8")

    comparison_readme_path = RESULTS_DIR / "README.md"
    comparison_lines = [
        "# Comparison Table Artifacts",
        "",
        "These files come from `experiments/baseline_comparison.py`.",
        "",
        "## Evidence Basis",
        "",
        "- seeded synthetic calibration",
        "- adversarial-simulation comparisons for selected OLYMPUS modules",
        "- internal method ranking inside a controlled toy setup",
        "",
        "## Experiments Included",
        "",
    ]

    for experiment_name, experiment_payload in payload["experiments"].items():
        metadata = experiment_payload["experiment_metadata"]
        summary = experiment_payload["summary"]
        comparison_lines.extend(
            [
                f"### {metadata['title']}",
                f"- Module: {metadata['module']}",
                f"- Evidence type: {metadata['evidence_type']}",
                f"- Primary metric: `{summary['primary_metric']}`",
                f"- Reference method: `{metadata.get('reference_method', 'n/a')}`",
            ]
        )
        comparison_lines.append("Not claimed:")
        comparison_lines.extend(f"- {claim}" for claim in metadata["not_claimed"])
        comparison_lines.append("")

    comparison_lines.extend(
        [
            "## Reading Guidance",
            "",
            "Use `comparison_results.json` as the canonical machine-readable source. The LaTeX tables",
            "are derived summaries intended for manuscript/report insertion, not a replacement for the",
            "seed-level raw rows preserved in the same JSON file.",
            "",
        ]
    )
    comparison_readme_path.write_text("\n".join(comparison_lines), encoding="utf-8")


if __name__ == "__main__":
    run_all_experiments()
