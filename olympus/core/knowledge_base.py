"""Shared knowledge base — persists across all modules."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from olympus.core.config import CONFIG
from olympus.core.logger import get_logger

log = get_logger("knowledge_base")


@dataclass
class ThreatRecord:
    id: str
    type: str                   # "vulnerability", "malware", "ttp", "ioc", "signature"
    name: str
    description: str
    severity: str               # "critical", "high", "medium", "low", "info"
    mitre_techniques: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_module: str = ""
    confidence: float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPattern:
    id: str
    name: str
    technique_id: str           # MITRE ATT&CK ID e.g. "T1059"
    tactic: str                 # e.g. "execution"
    description: str
    success_rate: float = 0.0
    detection_rate: float = 0.0
    evolution_generation: int = 0
    payload_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseRecord:
    id: str
    name: str
    targets_technique: str      # MITRE technique this counters
    description: str
    effectiveness: float = 0.0
    false_positive_rate: float = 0.0
    evolution_generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """Thread-safe, persistent shared knowledge store."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or CONFIG.knowledge_base.path
        self._lock = threading.RLock()
        self._threats: dict[str, ThreatRecord] = {}
        self._attack_patterns: dict[str, AttackPattern] = {}
        self._defenses: dict[str, DefenseRecord] = {}
        self._stats: dict[str, Any] = defaultdict(int)
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as fh:
                data = json.load(fh)
            for r in data.get("threats", []):
                t = ThreatRecord(**r)
                self._threats[t.id] = t
            for r in data.get("attack_patterns", []):
                p = AttackPattern(**r)
                self._attack_patterns[p.id] = p
            for r in data.get("defenses", []):
                d = DefenseRecord(**r)
                self._defenses[d.id] = d
            self._stats.update(data.get("stats", {}))
            log.info("Knowledge base loaded: %d threats, %d patterns, %d defenses",
                     len(self._threats), len(self._attack_patterns), len(self._defenses))
        except Exception as exc:
            log.warning("Failed to load knowledge base: %s", exc)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "threats": [asdict(t) for t in self._threats.values()],
                "attack_patterns": [asdict(p) for p in self._attack_patterns.values()],
                "defenses": [asdict(d) for d in self._defenses.values()],
                "stats": dict(self._stats),
            }
        with open(self._path, "w") as fh:
            json.dump(data, fh, indent=2)

    # ── threat records ────────────────────────────────────────────────────────

    def add_threat(self, record: ThreatRecord) -> None:
        with self._lock:
            existing = self._threats.get(record.id)
            if existing:
                existing.last_seen = time.time()
                existing.confidence = max(existing.confidence, record.confidence)
            else:
                self._threats[record.id] = record
                self._stats["total_threats"] += 1

    def get_threat(self, threat_id: str) -> Optional[ThreatRecord]:
        return self._threats.get(threat_id)

    def query_threats(self, type_filter: str | None = None,
                      severity: str | None = None,
                      tag: str | None = None,
                      limit: int = 100) -> list[ThreatRecord]:
        with self._lock:
            results = list(self._threats.values())
        if type_filter:
            results = [r for r in results if r.type == type_filter]
        if severity:
            results = [r for r in results if r.severity == severity]
        if tag:
            results = [r for r in results if tag in r.tags]
        results.sort(key=lambda r: r.last_seen, reverse=True)
        return results[:limit]

    # ── attack patterns ───────────────────────────────────────────────────────

    def add_attack_pattern(self, pattern: AttackPattern) -> None:
        with self._lock:
            self._attack_patterns[pattern.id] = pattern
            self._stats["total_patterns"] += 1

    def get_attack_patterns(self, tactic: str | None = None) -> list[AttackPattern]:
        with self._lock:
            patterns = list(self._attack_patterns.values())
        if tactic:
            patterns = [p for p in patterns if p.tactic == tactic]
        return sorted(patterns, key=lambda p: p.success_rate, reverse=True)

    # ── defenses ─────────────────────────────────────────────────────────────

    def add_defense(self, defense: DefenseRecord) -> None:
        with self._lock:
            self._defenses[defense.id] = defense
            self._stats["total_defenses"] += 1

    def get_defenses_for_technique(self, technique_id: str) -> list[DefenseRecord]:
        with self._lock:
            return [d for d in self._defenses.values()
                    if d.targets_technique == technique_id]

    # ── stats ─────────────────────────────────────────────────────────────────

    def increment_stat(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[key] += amount

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def summary(self) -> dict[str, int]:
        return {
            "threats": len(self._threats),
            "attack_patterns": len(self._attack_patterns),
            "defenses": len(self._defenses),
        }


KB = KnowledgeBase()
