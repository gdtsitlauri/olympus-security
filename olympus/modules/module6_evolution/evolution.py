"""Module 6 — Self-Evolution (OLYMPUS-TITAN co-evolutionary loop)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import AttackPattern, DefenseRecord
from olympus.modules.module6_evolution.titan import TITANConfig, TITANEngine, GenerationStats


class SelfEvolutionModule(BaseModule):
    MODULE_ID = "module6_evolution"
    MODULE_NAME = "Self-Evolution (OLYMPUS-TITAN)"
    MODULE_TYPE = "core"

    def __init__(self) -> None:
        super().__init__()
        self._engine: Optional[TITANEngine] = None

    def run(
        self,
        generations: int = 50,
        population_size: int = 30,
        gene_dim: int = 16,
        seed: int = 42,
        save_results: bool = True,
        **kwargs: Any,
    ) -> ModuleResult:
        result, t0 = self._start_result()

        config = TITANConfig(
            population_size=population_size,
            gene_dim=gene_dim,
            generations=generations,
        )
        self._engine = TITANEngine(config=config, seed=seed)
        self._engine.initialize()

        self.log.info("Starting TITAN co-evolution: %d generations, pop=%d",
                      generations, population_size)

        gen_log: list[dict] = []

        def on_gen(stats: GenerationStats) -> None:
            gen_log.append({
                "generation": stats.generation,
                "attack_best": round(stats.attack_best_fitness, 4),
                "attack_mean": round(stats.attack_mean_fitness, 4),
                "defense_best": round(stats.defense_best_fitness, 4),
                "defense_mean": round(stats.defense_mean_fitness, 4),
                "gap": round(stats.attack_defense_gap, 4),
            })

        history = self._engine.evolve(callback=on_gen)

        # ── store best strategies to KB ───────────────────────────────────────
        best_attack = self._engine.best_attack()
        best_defense = self._engine.best_defense()

        if best_attack:
            self.kb.add_attack_pattern(AttackPattern(
                id=f"titan-atk-{best_attack.id}",
                name=best_attack.name,
                technique_id="T1059",
                tactic="execution",
                description=f"TITAN-evolved attack strategy (gen {generations})",
                success_rate=round(best_attack.fitness, 4),
                evolution_generation=generations,
                metadata={"genes": best_attack.genes[:4]},
            ))

        if best_defense:
            self.kb.add_defense(DefenseRecord(
                id=f"titan-def-{best_defense.id}",
                name=best_defense.name,
                targets_technique="T1059",
                description=f"TITAN-evolved defense strategy (gen {generations})",
                effectiveness=round(best_defense.fitness, 4),
                evolution_generation=generations,
                metadata={"genes": best_defense.genes[:4]},
            ))

        # ── findings ──────────────────────────────────────────────────────────
        result.add_finding(
            severity="info",
            title="TITAN evolution complete",
            detail=(
                f"Best attack fitness: {best_attack.fitness:.4f} | "
                f"Best defense fitness: {best_defense.fitness:.4f} | "
                f"Convergence: {self._engine.convergence_score():.4f}"
            ) if best_attack and best_defense else "Evolution ran but no strategies found",
            generations=generations,
            best_attack_id=best_attack.id if best_attack else None,
            best_defense_id=best_defense.id if best_defense else None,
        )

        # Check for dominant attack (security concern)
        if best_attack and best_attack.fitness > 0.8:
            result.add_finding(
                severity="high",
                title="Attack population dominates — defense needs improvement",
                detail=f"Attack win rate {best_attack.fitness:.1%} > 80% — deploy stronger defenses",
            )
        elif best_defense and best_defense.fitness > 0.8:
            result.add_finding(
                severity="info",
                title="Defense population dominates — system well-defended",
                detail=f"Defense win rate {best_defense.fitness:.1%}",
            )

        # ── save results ──────────────────────────────────────────────────────
        if save_results:
            out_path = Path("results") / "titan_evolution.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "config": {
                        "generations": generations,
                        "population_size": population_size,
                        "gene_dim": gene_dim,
                        "seed": seed,
                    },
                    "history": gen_log,
                    "final": {
                        "best_attack_fitness": best_attack.fitness if best_attack else 0,
                        "best_defense_fitness": best_defense.fitness if best_defense else 0,
                        "convergence": self._engine.convergence_score(),
                    },
                }, f, indent=2)

        result.metrics = {
            "generations_run": generations,
            "population_size": population_size,
            "best_attack_fitness": round(best_attack.fitness, 4) if best_attack else 0.0,
            "best_defense_fitness": round(best_defense.fitness, 4) if best_defense else 0.0,
            "convergence_score": self._engine.convergence_score(),
            "total_evaluations": generations * population_size * config.evaluations_per_gen,
        }

        self.log.info("TITAN complete: atk=%.4f def=%.4f convergence=%.4f",
                      best_attack.fitness if best_attack else 0,
                      best_defense.fitness if best_defense else 0,
                      self._engine.convergence_score())

        return self._finish_result(result, t0)

    def get_evolution_history(self) -> list[GenerationStats]:
        return self._engine.history if self._engine else []
