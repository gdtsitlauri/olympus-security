"""OLYMPUS-TITAN — Total Intelligent Threat Analysis and Neutralization.

Co-evolutionary loop where attack strategies and defense strategies evolve
together as two competing populations, each improving against the other.

Algorithm:
  1. Initialize attack population A and defense population D
  2. Evaluate: each attack strategy plays against each defense strategy
  3. Select top performers (tournament selection)
  4. Crossover + Mutation to produce offspring
  5. Reinforcement: strategies that succeed get higher fitness
  6. Repeat from step 2 until convergence or max generations
"""

from __future__ import annotations

import copy
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from olympus.core.logger import get_logger

log = get_logger("module6.titan")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False
    nn = type("_TorchNNFallback", (), {"Module": object})()  # type: ignore[assignment]


# ── Strategy representation ───────────────────────────────────────────────────

@dataclass
class Strategy:
    """A parameterized security strategy (attack or defense)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_type: str = "attack"       # "attack" or "defense"
    name: str = ""
    genes: list[float] = field(default_factory=list)   # continuous parameter vector
    fitness: float = 0.0
    generation: int = 0
    wins: int = 0
    losses: int = 0
    lineage: list[str] = field(default_factory=list)   # parent IDs

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    def clone(self) -> "Strategy":
        c = copy.deepcopy(self)
        c.id = str(uuid.uuid4())[:8]
        c.lineage = [self.id] + self.lineage[:4]
        c.wins = 0
        c.losses = 0
        return c


# ── Neural fitness evaluator ──────────────────────────────────────────────────

class FitnessNetwork(nn.Module):  # type: ignore[misc]
    """Neural network that predicts attack success probability given (attack_genes, defense_genes)."""

    def __init__(self, gene_dim: int = 16) -> None:
        super().__init__()
        input_dim = gene_dim * 2   # concatenate attack + defense genes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, attack_genes, defense_genes):
        x = torch.cat([attack_genes, defense_genes], dim=-1)
        return self.net(x).squeeze(-1)


# ── TITAN evolutionary engine ─────────────────────────────────────────────────

@dataclass
class TITANConfig:
    population_size: int = 50
    gene_dim: int = 16
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    elite_fraction: float = 0.1
    evaluations_per_gen: int = 10        # how many matchups per strategy per gen
    fitness_lr: float = 1e-3


@dataclass
class GenerationStats:
    generation: int
    attack_best_fitness: float
    attack_mean_fitness: float
    defense_best_fitness: float
    defense_mean_fitness: float
    attack_defense_gap: float
    timestamp: float = field(default_factory=time.time)


class TITANEngine:
    """Co-evolutionary attack-defense optimization engine."""

    def __init__(self, config: TITANConfig | None = None, seed: int = 42) -> None:
        self.config = config or TITANConfig()
        self._rng = random.Random(seed)
        self._device = None
        self._fitness_net: Optional[FitnessNetwork] = None
        self._fitness_optimizer = None

        if _TORCH:
            from olympus.core.device import get_device
            self._device = get_device()
            self._fitness_net = FitnessNetwork(self.config.gene_dim).to(self._device)
            self._fitness_optimizer = optim.Adam(
                self._fitness_net.parameters(), lr=self.config.fitness_lr
            )
            log.info("TITAN fitness network on %s", self._device)

        self.attack_population: list[Strategy] = []
        self.defense_population: list[Strategy] = []
        self.history: list[GenerationStats] = []

    # ── initialization ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        n = self.config.population_size
        d = self.config.gene_dim
        self.attack_population = [
            Strategy(
                strategy_type="attack",
                name=f"Attack-{i}",
                genes=[self._rng.gauss(0, 1) for _ in range(d)],
            )
            for i in range(n)
        ]
        self.defense_population = [
            Strategy(
                strategy_type="defense",
                name=f"Defense-{i}",
                genes=[self._rng.gauss(0, 1) for _ in range(d)],
            )
            for i in range(n)
        ]
        log.info("TITAN initialized: %d attack, %d defense strategies", n, n)

    # ── evaluation ────────────────────────────────────────────────────────────

    def _evaluate_neural(self, attack: Strategy, defense: Strategy) -> float:
        """Neural fitness: probability that attack beats defense."""
        import torch
        a = torch.tensor(attack.genes, dtype=torch.float32).unsqueeze(0).to(self._device)
        d = torch.tensor(defense.genes, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            prob = self._fitness_net(a, d).item()
        return prob

    def _evaluate_heuristic(self, attack: Strategy, defense: Strategy) -> float:
        """Dot-product similarity heuristic: attack wins if genes oppose defense."""
        a = attack.genes
        d = defense.genes
        dot = sum(ai * di for ai, di in zip(a, d))
        norm_a = math.sqrt(sum(x**2 for x in a)) or 1e-6
        norm_d = math.sqrt(sum(x**2 for x in d)) or 1e-6
        cosine = dot / (norm_a * norm_d)
        # Attack wins when strategies are opposed (cosine ≈ -1)
        return (1 - cosine) / 2.0

    def _evaluate(self, attack: Strategy, defense: Strategy) -> float:
        if _TORCH and self._fitness_net:
            return self._evaluate_neural(attack, defense)
        return self._evaluate_heuristic(attack, defense)

    def _run_evaluations(self) -> list[tuple[float, float]]:
        """Run random matchups, return (attack_prob, defense_prob) pairs."""
        k = self.config.evaluations_per_gen
        n = len(self.attack_population)
        results = []
        for _ in range(n * k):
            a = self._rng.choice(self.attack_population)
            d = self._rng.choice(self.defense_population)
            p = self._evaluate(a, d)
            if p > 0.5:
                a.wins += 1; d.losses += 1
            else:
                a.losses += 1; d.wins += 1
            # Fitness = running win rate
            a.fitness = a.win_rate
            d.fitness = 1 - a.win_rate  # defense wins when attack loses
            results.append((p, 1 - p))
        return results

    # ── selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population: list[Strategy]) -> Strategy:
        k = min(self.config.tournament_size, len(population))
        competitors = self._rng.sample(population, k)
        return max(competitors, key=lambda s: s.fitness)

    def _elites(self, population: list[Strategy]) -> list[Strategy]:
        n_elite = max(1, int(len(population) * self.config.elite_fraction))
        return sorted(population, key=lambda s: s.fitness, reverse=True)[:n_elite]

    # ── genetic operators ─────────────────────────────────────────────────────

    def _crossover(self, a: Strategy, b: Strategy) -> Strategy:
        child = a.clone()
        if self._rng.random() < self.config.crossover_rate:
            point = self._rng.randint(1, len(a.genes) - 1)
            child.genes = a.genes[:point] + b.genes[point:]
        return child

    def _mutate(self, s: Strategy, gen: int) -> Strategy:
        child = s.clone()
        child.generation = gen
        # Adaptive mutation strength decays over generations
        strength = self.config.mutation_strength * math.exp(-gen / 200)
        for i in range(len(child.genes)):
            if self._rng.random() < self.config.mutation_rate:
                child.genes[i] += self._rng.gauss(0, strength)
                child.genes[i] = max(-5.0, min(5.0, child.genes[i]))  # clamp
        return child

    # ── fitness network training ───────────────────────────────────────────────

    def _update_fitness_net(self, matchups: list[tuple[Strategy, Strategy, float]]) -> float:
        if not _TORCH or not self._fitness_net or not matchups:
            return 0.0
        import torch

        a_batch = torch.tensor(
            [m[0].genes for m in matchups], dtype=torch.float32
        ).to(self._device)
        d_batch = torch.tensor(
            [m[1].genes for m in matchups], dtype=torch.float32
        ).to(self._device)
        labels = torch.tensor(
            [m[2] for m in matchups], dtype=torch.float32
        ).to(self._device)

        self._fitness_optimizer.zero_grad()
        preds = self._fitness_net(a_batch, d_batch)
        loss = nn.functional.binary_cross_entropy(preds, labels)
        loss.backward()
        self._fitness_optimizer.step()
        return loss.item()

    # ── main loop ─────────────────────────────────────────────────────────────

    def evolve(
        self,
        n_generations: Optional[int] = None,
        callback: Optional[Callable[[GenerationStats], None]] = None,
    ) -> list[GenerationStats]:
        if not self.attack_population:
            self.initialize()

        n_gen = n_generations or self.config.generations

        for gen in range(1, n_gen + 1):
            # Reset per-generation win/loss counters
            for s in self.attack_population + self.defense_population:
                s.wins = 0; s.losses = 0

            # Evaluate
            matchup_data = []
            for _ in range(self.config.evaluations_per_gen * len(self.attack_population)):
                a = self._rng.choice(self.attack_population)
                d = self._rng.choice(self.defense_population)
                p = self._evaluate(a, d)
                result = 1 if p > 0.5 else 0
                a.wins += result; d.wins += 1 - result
                a.losses += 1 - result; d.losses += result
                matchup_data.append((a, d, float(result)))

            # Update fitness
            for s in self.attack_population:
                s.fitness = s.win_rate
            for s in self.defense_population:
                s.fitness = s.win_rate

            # Train fitness network
            if gen % 5 == 0:
                self._update_fitness_net(matchup_data[-100:])

            # Record stats
            atk_fits = [s.fitness for s in self.attack_population]
            def_fits = [s.fitness for s in self.defense_population]
            stats = GenerationStats(
                generation=gen,
                attack_best_fitness=max(atk_fits),
                attack_mean_fitness=sum(atk_fits) / len(atk_fits),
                defense_best_fitness=max(def_fits),
                defense_mean_fitness=sum(def_fits) / len(def_fits),
                attack_defense_gap=max(atk_fits) - max(def_fits),
            )
            self.history.append(stats)

            if callback:
                callback(stats)

            if gen % 10 == 0:
                log.info("Gen %d | Atk best=%.3f | Def best=%.3f | Gap=%.3f",
                         gen, stats.attack_best_fitness, stats.defense_best_fitness,
                         stats.attack_defense_gap)

            # Evolve next generation
            self.attack_population = self._next_gen(self.attack_population, gen)
            self.defense_population = self._next_gen(self.defense_population, gen)

        return self.history

    def _next_gen(self, population: list[Strategy], gen: int) -> list[Strategy]:
        n = len(population)
        elites = self._elites(population)
        new_pop = [s.clone() for s in elites]

        while len(new_pop) < n:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child, gen)
            new_pop.append(child)

        return new_pop[:n]

    # ── introspection ─────────────────────────────────────────────────────────

    def best_attack(self) -> Optional[Strategy]:
        return max(self.attack_population, key=lambda s: s.fitness) \
               if self.attack_population else None

    def best_defense(self) -> Optional[Strategy]:
        return max(self.defense_population, key=lambda s: s.fitness) \
               if self.defense_population else None

    def convergence_score(self) -> float:
        """0 = not converged, 1 = fully converged (no improvement)."""
        if len(self.history) < 10:
            return 0.0
        recent = self.history[-10:]
        atk_range = max(s.attack_best_fitness for s in recent) - \
                    min(s.attack_best_fitness for s in recent)
        return round(1 - min(atk_range * 10, 1.0), 4)
