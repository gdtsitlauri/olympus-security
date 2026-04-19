"""Module 3 — Zero-Day Discovery (AI-guided fuzzing + static analysis)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.modules.module3_zeroday.fuzzer import MutationEngine, NeuralSeedGenerator, TargetExecutor, CrashResult
from olympus.modules.module3_zeroday.static_analyzer import analyze_file, StaticFinding

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _cvss_score(crash: CrashResult) -> float:
    """Estimate CVSS 3.1 base score from crash characteristics."""
    scores = {"critical": 9.1, "high": 7.5, "medium": 5.3, "low": 3.0, "unknown": 4.0}
    base = scores.get(crash.crash_type, 4.0)
    if crash.crash_type in ("segfault", "abort"):
        base = max(base, 8.0)   # memory corruption — likely exploitable
    return round(base, 1)


class ZeroDayModule(BaseModule):
    MODULE_ID = "module3_zeroday"
    MODULE_NAME = "Zero-Day Discovery"
    MODULE_TYPE = "offensive"

    def __init__(self) -> None:
        super().__init__()
        self._mutation_engine = MutationEngine()
        self._neural_gen: Optional[NeuralSeedGenerator] = None
        self._device = None
        if _TORCH:
            from olympus.core.device import get_device
            self._device = get_device()
            self._neural_gen = NeuralSeedGenerator().to(self._device)
            self._neural_gen.eval()

    def run(
        self,
        mode: str = "static",               # "static", "fuzz", "both"
        target_path: str = ".",             # path to analyze (static)
        target_cmd: Optional[list[str]] = None,  # command to fuzz
        fuzz_iterations: int = 1000,
        seed_corpus: Optional[list[bytes]] = None,
        use_neural_seeds: bool = True,
        **kwargs: Any,
    ) -> ModuleResult:
        result, t0 = self._start_result()

        static_findings: list[StaticFinding] = []
        crash_results: list[CrashResult] = []

        # ── static analysis ───────────────────────────────────────────────────
        if mode in ("static", "both"):
            root = Path(target_path)
            files = []
            if root.is_file():
                files = [root]
            else:
                for ext in (".py", ".c", ".cpp", ".h", ".cc", ".js", ".php", ".rb"):
                    files.extend(root.rglob(f"*{ext}"))

            self.log.info("Static analysis: %d files", len(files))
            for f in files:
                try:
                    findings = analyze_file(f)
                    static_findings.extend(findings)
                except Exception as exc:
                    self.log.debug("Static analysis error %s: %s", f, exc)

        # ── fuzzing ───────────────────────────────────────────────────────────
        if mode in ("fuzz", "both") and target_cmd:
            executor = TargetExecutor(target_cmd)
            corpus = list(seed_corpus or [])

            # Generate initial seeds
            if not corpus:
                corpus = self._mutation_engine.generate_seeds(20)

            if use_neural_seeds and self._neural_gen is not None:
                corpus.extend(self._neural_seeds(32))

            self.log.info("Fuzzing %s with %d seeds, %d iterations",
                          " ".join(target_cmd), len(corpus), fuzz_iterations)

            for i in range(fuzz_iterations):
                seed = corpus[i % len(corpus)]
                strategies = ["bit_flip", "byte_flip", "interesting_values",
                              "arithmetic", "splice", "havoc", "dictionary"]
                strategy = strategies[i % len(strategies)]
                mutated = self._mutation_engine.mutate(seed, strategy)

                crash = executor.execute(mutated)
                if crash:
                    crash_results.append(crash)
                    self.log.info("CRASH #%d: %s (signal=%s)", len(crash_results),
                                  crash.crash_type, crash.signal)
                    # Add interesting input to corpus
                    corpus.append(mutated)
                    if len(corpus) > 500:
                        corpus.pop(0)

        # ── compile results ───────────────────────────────────────────────────
        # Static findings
        for sf in static_findings:
            result.add_finding(
                severity=sf.severity,
                title=sf.title,
                detail=f"{sf.file}:{sf.line} — {sf.snippet}",
                rule_id=sf.rule_id,
                cwe=sf.cwe,
                category=sf.category,
            )
            self.kb.add_threat(ThreatRecord(
                id=f"zeroday-static-{sf.rule_id}-{hash(sf.file + str(sf.line)) % 10000}",
                type="vulnerability",
                name=sf.title,
                description=f"Static analysis finding: {sf.rule_id} in {sf.file}:{sf.line}",
                severity=sf.severity,
                source_module=self.MODULE_ID,
                confidence=0.75,
                metadata={"cwe": sf.cwe, "file": sf.file, "line": sf.line},
            ))

        # Crash findings
        for crash in crash_results:
            cvss = _cvss_score(crash)
            severity = "critical" if cvss >= 9.0 else "high" if cvss >= 7.0 else "medium"
            result.add_finding(
                severity=severity,
                title=f"Fuzzing crash: {crash.crash_type}",
                detail=crash.to_poc(),
                crash_id=crash.crash_id,
                signal=crash.signal,
                cvss=cvss,
            )
            self.kb.add_threat(ThreatRecord(
                id=f"zeroday-crash-{crash.crash_id}",
                type="vulnerability",
                name=f"Zero-day crash: {crash.crash_type}",
                description=crash.backtrace[:500],
                severity=severity,
                source_module=self.MODULE_ID,
                confidence=0.9,
                metadata={"cvss": cvss, "crash_type": crash.crash_type},
            ))

        result.metrics = {
            "static_findings": len(static_findings),
            "crashes_found": len(crash_results),
            "unique_crashes": len(crash_results),
            "critical_findings": sum(1 for f in result.findings if f["severity"] == "critical"),
            "fuzz_iterations": fuzz_iterations if mode in ("fuzz", "both") else 0,
        }

        self.log.info("Zero-day discovery: %d static findings, %d crashes",
                      len(static_findings), len(crash_results))
        return self._finish_result(result, t0)

    def _neural_seeds(self, n: int) -> list[bytes]:
        if not _TORCH or self._neural_gen is None:
            return []
        try:
            import torch
            samples = self._neural_gen.sample(n).cpu()
            result = []
            for s in samples:
                b = (s.numpy() * 255).astype("uint8").tobytes()
                result.append(b)
            return result
        except Exception as exc:
            self.log.debug("Neural seed generation failed: %s", exc)
            return []
