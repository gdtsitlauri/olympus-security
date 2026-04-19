"""Module 9 — LLM Defense (jailbreak detection, prompt injection hardening)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.core.logger import get_logger

log = get_logger("module9")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ── Jailbreak pattern taxonomy ────────────────────────────────────────────────

_JAILBREAK_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # Role confusion / DAN variants
    ("role_confusion", re.compile(
        r'\b(dan|do anything now|jailbreak|ignore (previous|all|your) (instructions?|rules?|'
        r'guidelines?)|pretend (you are|to be)|act as (if you are|a|an)|roleplay|'
        r'simulation mode|developer mode|unrestricted mode)\b', re.I
    ), 0.9),
    # Instruction override
    ("instruction_override", re.compile(
        r'\b(ignore|disregard|forget|override|bypass|circumvent|disable)'
        r'.*?(restriction|filter|safety|guard|policy|rule|instruction|limit)\b', re.I
    ), 0.85),
    # Prompt injection
    ("prompt_injection", re.compile(
        r'<\|?(system|user|assistant|human|ai)\|?>|'
        r'\[\[(system|inst|instruction)\]\]|'
        r'###\s*(system|instruction)|'
        r'<\s*system\s*>|'
        r'<<SYS>>|'
        r'\[INST\]', re.I
    ), 0.95),
    # Encoding evasion
    ("encoding_evasion", re.compile(
        r'(?:base64|rot13|hex|unicode|morse|binary)\s*(?:decode|encode|translate|convert)', re.I
    ), 0.7),
    # Hypothetical framing
    ("hypothetical_frame", re.compile(
        r'\b(hypothetically|theoretically|in fiction|for a (story|novel|book|movie|game)|'
        r'imagine (that|if)|let.?s say|what if|suppose)\b.*?\b(how to|instructions?|steps?)\b',
        re.I | re.DOTALL,
    ), 0.65),
    # Token manipulation
    ("token_manipulation", re.compile(
        r'(\w)\s+\1\s+\1|'          # repeated spaced chars
        r'[\u200b\u200c\u200d\ufeff]|'  # zero-width chars
        r'(?:[a-z]-){3,}',          # letter-dash sequences
        re.I,
    ), 0.6),
    # Goal hijacking
    ("goal_hijacking", re.compile(
        r'\b(your (real|true|actual|hidden|secret) (purpose|goal|task|mission|objective|instruction)|'
        r'you (must|should|will|shall) (ignore|forget|override|reveal)|'
        r'new (instructions?|rules?|system|prompt))\b', re.I
    ), 0.8),
    # Harmful content extraction
    ("harmful_extraction", re.compile(
        r'\b(how to (make|create|build|synthesize|produce)|'
        r'step[- ]by[- ]step (instructions?|guide) (for|to)|'
        r'detailed (instructions?|guide|tutorial) (on|for|to))\b.*?'
        r'\b(weapon|bomb|drug|malware|exploit|hack|poison|toxin)\b',
        re.I | re.DOTALL,
    ), 0.99),
]

# ── Defense strategies ────────────────────────────────────────────────────────

_SANITIZATION_RULES: list[tuple[re.Pattern, str]] = [
    # Remove prompt injection markers
    (re.compile(r'<\|?(system|user|assistant)\|?>', re.I), "[REMOVED]"),
    (re.compile(r'\[\[(system|inst)\]\]', re.I), "[REMOVED]"),
    (re.compile(r'<<SYS>>|\[INST\]', re.I), "[REMOVED]"),
    # Remove zero-width chars
    (re.compile(r'[\u200b\u200c\u200d\ufeff]'), ""),
    # Normalize whitespace manipulation
    (re.compile(r'(\w)\s{2,}(\w)'), r'\1 \2'),
]


@dataclass
class PromptAnalysis:
    text: str
    is_jailbreak: bool
    confidence: float
    risk_score: float           # 0-100
    detected_patterns: list[tuple[str, float]]   # [(pattern_name, confidence)]
    sanitized_text: str
    recommended_action: str     # "allow", "sanitize", "block", "review"


class JailbreakDetector:
    """Multi-strategy LLM jailbreak and prompt injection detector."""

    def __init__(self) -> None:
        self._classifier: Optional[nn.Module] = None
        self._device = None
        if _TORCH:
            from olympus.core.device import get_device
            self._device = get_device()
            self._classifier = self._build_classifier().to(self._device)
            self._classifier.eval()
            log.info("Jailbreak classifier on %s", self._device)

    def _build_classifier(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(len(_JAILBREAK_PATTERNS) + 5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),     # binary: clean / jailbreak
        )

    def _pattern_features(self, text: str) -> list[float]:
        feats = []
        for name, pattern, weight in _JAILBREAK_PATTERNS:
            matches = len(pattern.findall(text))
            feats.append(min(matches * weight, 1.0))

        # Additional structural features
        feats.append(min(len(text) / 2000.0, 1.0))        # text length
        feats.append(float(text.count("\n") > 10))         # many newlines
        feats.append(min(text.count("```") / 4.0, 1.0))   # code blocks
        feats.append(float(bool(re.search(r'<[a-z]+>', text, re.I))))  # HTML tags
        feats.append(min(len(set(text.split())) / 100.0, 1.0))  # vocab diversity
        return feats

    def analyze(self, text: str) -> PromptAnalysis:
        detected = []
        for name, pattern, base_confidence in _JAILBREAK_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                conf = min(base_confidence * len(matches), 1.0)
                detected.append((name, round(conf, 3)))

        # Sanitize
        sanitized = text
        for rule, replacement in _SANITIZATION_RULES:
            sanitized = rule.sub(replacement, sanitized)

        # Compute risk score
        pattern_score = sum(c for _, c in detected)
        if _TORCH and self._classifier:
            feats = self._pattern_features(text)
            import torch
            x = torch.tensor([feats], dtype=torch.float32).to(self._device)
            with torch.no_grad():
                logits = self._classifier(x)
                prob_jb = torch.softmax(logits, dim=-1)[0, 1].item()
            risk = min(prob_jb * 100 + pattern_score * 10, 100.0)
        else:
            risk = min(pattern_score * 20, 100.0)

        is_jailbreak = risk > 40 or any(n in ("prompt_injection", "harmful_extraction")
                                         for n, _ in detected)
        confidence = min(0.5 + risk / 200, 0.99)

        if risk > 70:
            action = "block"
        elif risk > 40:
            action = "sanitize"
        elif risk > 20:
            action = "review"
        else:
            action = "allow"

        return PromptAnalysis(
            text=text,
            is_jailbreak=is_jailbreak,
            confidence=round(confidence, 4),
            risk_score=round(risk, 1),
            detected_patterns=detected,
            sanitized_text=sanitized,
            recommended_action=action,
        )

    def batch_analyze(self, texts: list[str]) -> list[PromptAnalysis]:
        return [self.analyze(t) for t in texts]


class LLMDefenseModule(BaseModule):
    MODULE_ID = "module9_llm_defense"
    MODULE_NAME = "LLM Defense & Jailbreak Detection"
    MODULE_TYPE = "defensive"

    def __init__(self) -> None:
        super().__init__()
        self._detector = JailbreakDetector()

    def run(
        self,
        mode: str = "detect",
        prompts: list[str] | None = None,
        prompt_file: Optional[str] = None,
        **kwargs: Any,
    ) -> ModuleResult:
        """
        Modes:
          detect    — analyze provided prompts for jailbreak patterns
          benchmark — evaluate all defenses against published benchmark dataset
          full      — benchmark + detect
        """
        result, t0 = self._start_result()

        if mode in ("benchmark", "full"):
            self._run_benchmark_evaluation(result)

        if mode in ("detect", "full"):
            self._run_detection(result, prompts or [], prompt_file)

        return self._finish_result(result, t0)

    # ── Benchmark evaluation ──────────────────────────────────────────────────

    def _run_benchmark_evaluation(self, result: ModuleResult) -> None:
        from pathlib import Path
        from olympus.modules.module9_llm_defense.benchmark_dataset import (
            load_benchmark, assert_no_harmful_generation,
        )
        from olympus.modules.module9_llm_defense.defense_evaluator import evaluate_defenses

        assert_no_harmful_generation()
        samples = load_benchmark(
            output_path=Path("data/samples/jailbreak_benchmark.csv")
        )

        eval_report = evaluate_defenses(
            samples=samples,
            output_csv=Path("results/comparison_tables/module9_results.csv"),
            report_path=Path("results/module9/hardening_report.md"),
        )

        result.add_finding(
            severity="info",
            title=f"Benchmark loaded: {eval_report.n_samples} samples "
                  f"({eval_report.n_jailbreak} jailbreak, {eval_report.n_benign} benign)",
            detail="Sources: JailbreakBench / AdvBench / published literature",
            n_samples=eval_report.n_samples,
            n_jailbreak=eval_report.n_jailbreak,
            n_benign=eval_report.n_benign,
        )

        best = min(eval_report.results, key=lambda r: r.asr)
        for dr in eval_report.results:
            result.add_finding(
                severity="info",
                title=f"Defense '{dr.defense_name}': ASR={dr.asr:.3f} FPR={dr.fpr:.3f} F1={dr.f1:.3f}",
                detail=(
                    f"TP={dr.true_positives} FP={dr.false_positives} "
                    f"TN={dr.true_negatives} FN={dr.false_negatives}"
                ),
                defense=dr.defense_name,
                asr=round(dr.asr, 4),
                fpr=round(dr.fpr, 4),
                precision=round(dr.precision, 4),
                recall=round(dr.recall, 4),
                f1=round(dr.f1, 4),
            )

        result.add_finding(
            severity="info",
            title=f"Best defense: '{best.defense_name}' (ASR={best.asr:.3f})",
            detail=eval_report.summary_table(),
        )

        result.metrics = {
            "benchmark_samples": eval_report.n_samples,
            "best_defense_asr": round(best.asr, 4),
            "best_defense_fpr": round(best.fpr, 4),
            "best_defense_f1": round(best.f1, 4),
            "ensemble_asr": round(
                next(r.asr for r in eval_report.results if r.defense_name == "ensemble"), 4),
            "ensemble_fpr": round(
                next(r.fpr for r in eval_report.results if r.defense_name == "ensemble"), 4),
        }

    # ── Detection mode ────────────────────────────────────────────────────────

    def _run_detection(
        self,
        result: ModuleResult,
        prompts: list[str],
        prompt_file: Optional[str],
    ) -> None:
        from pathlib import Path

        texts = list(prompts)
        if prompt_file:
            p = Path(prompt_file)
            if p.exists():
                texts.extend(p.read_text().splitlines())

        if not texts:
            result.add_finding("info", "No prompts to analyze",
                               "Provide prompts or prompt_file")
            return

        analyses = self._detector.batch_analyze(texts)

        for analysis in analyses:
            if analysis.is_jailbreak:
                severity = "critical" if analysis.risk_score > 70 else "high"
                result.add_finding(
                    severity=severity,
                    title=f"Jailbreak/injection detected (risk={analysis.risk_score:.0f}/100)",
                    detail=f"Action: {analysis.recommended_action} | "
                           f"Patterns: {[p[0] for p in analysis.detected_patterns]}",
                    detected_patterns=analysis.detected_patterns,
                    recommended_action=analysis.recommended_action,
                    sanitized_preview=analysis.sanitized_text[:200],
                )
                self.kb.add_threat(ThreatRecord(
                    id=f"llm-jb-{hash(analysis.text[:100]) % 100000}",
                    type="ttp",
                    name="LLM jailbreak attempt",
                    description=f"Patterns: {[p[0] for p in analysis.detected_patterns]}",
                    severity=severity,
                    mitre_techniques=["T1059"],
                    source_module=self.MODULE_ID,
                    confidence=analysis.confidence,
                ))

        blocked = sum(1 for a in analyses if a.recommended_action == "block")
        sanitized = sum(1 for a in analyses if a.recommended_action == "sanitize")

        if not result.metrics:
            result.metrics = {}
        result.metrics.update({
            "prompts_analyzed": len(analyses),
            "jailbreaks_detected": sum(1 for a in analyses if a.is_jailbreak),
            "blocked": blocked,
            "sanitized": sanitized,
            "allowed": sum(1 for a in analyses if a.recommended_action == "allow"),
            "avg_risk_score": round(
                sum(a.risk_score for a in analyses) / max(len(analyses), 1), 2),
        })
