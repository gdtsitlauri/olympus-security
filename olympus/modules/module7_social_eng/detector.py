"""Module 7 — Social Engineering Detection + Research Framework.

Combines:
- Synthetic phishing dataset generation (in-memory, no network)
- ML-based detection (Random Forest + GPU LSTM)
- Awareness training simulator
- Incoming SE detection (production use)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.core.logger import get_logger
from olympus.modules.module7_social_eng.synthetic_dataset import (
    SyntheticEmail, generate_dataset, assert_no_network_calls
)
from olympus.modules.module7_social_eng.phishing_ml import (
    extract_features, prepare_features, train_rf, train_lstm,
    save_models, TrainingMetrics, FEATURE_DIM,
)
from olympus.modules.module7_social_eng.awareness_trainer import AwarenessTrainer

import re

log = get_logger("module7")

# ── Incoming SE detection (production) ───────────────────────────────────────

_URGENCY = re.compile(r'\b(urgent|immediately|act now|expires?|deadline|limited time|within \d+ hours?|asap)\b', re.I)
_AUTHORITY = re.compile(r'\b(ceo|president|it department|security team|irs|fbi|bank|billing|noreply)\b', re.I)
_FEAR = re.compile(r'\b(suspend|compromised|unauthorized|unusual activity|locked|blocked|legal action)\b', re.I)
_REWARD = re.compile(r'\b(winner|won|prize|reward|gift card|congratulations|selected|free)\b', re.I)
_URL_PATTERN = re.compile(r'https?://[^\s<>"\']+', re.I)
_LOOKALIKE = re.compile(r'(paypa[^l]|amaz[^o]n|app1e|micros0ft|g00gle|faceb[^o]ok)', re.I)
_SE_CLASSES = ["legitimate", "suspicious", "phishing", "scam"]

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


class SocialEngDetectionModule(BaseModule):
    MODULE_ID = "module7_social_eng"
    MODULE_NAME = "Social Engineering Detection"
    MODULE_TYPE = "defensive"

    def __init__(self) -> None:
        super().__init__()
        self._rf_model = None
        self._lstm_model = None
        self._scaler = None
        self._dataset: Optional[list[SyntheticEmail]] = None
        self._awareness_trainer: Optional[AwarenessTrainer] = None

    def run(
        self,
        mode: str = "detect",
        texts: list[str] | None = None,
        email_files: list[str] | None = None,
        n_phishing: int = 5000,
        n_legit: int = 5000,
        epochs: int = 20,
        seed: int = 42,
        **kwargs: Any,
    ) -> ModuleResult:
        """
        Modes:
          detect   — detect SE in provided texts/emails (production)
          research — generate dataset + train detector
          awareness — run awareness training simulation
          full     — research + awareness
        """
        result, t0 = self._start_result()

        if mode in ("research", "full"):
            self._run_research(result, n_phishing, n_legit, epochs, seed)

        if mode in ("detect",):
            self._run_detection(result, texts or [], email_files or [])

        if mode in ("awareness", "full"):
            self._run_awareness(result, seed)

        return self._finish_result(result, t0)

    # ── Research mode ─────────────────────────────────────────────────────────

    def _run_research(
        self, result: ModuleResult,
        n_phishing: int, n_legit: int, epochs: int, seed: int,
    ) -> None:
        self.log.info("Generating %d phishing + %d legit synthetic emails...",
                      n_phishing, n_legit)

        # SAFETY: verify no network calls
        assert_no_network_calls()

        dataset_path = Path("data/samples/phishing_dataset.csv")
        self._dataset = generate_dataset(
            n_phishing=n_phishing, n_legit=n_legit, seed=seed,
            output_path=dataset_path,
        )
        self.log.info("Dataset saved to %s (%d samples)", dataset_path, len(self._dataset))

        result.add_finding(
            severity="info",
            title=f"Synthetic dataset generated: {len(self._dataset)} samples",
            detail=f"Phishing: {n_phishing} | Legit: {n_legit} | Path: {dataset_path}",
            dataset_path=str(dataset_path),
        )

        if not _NP:
            result.add_finding("medium", "numpy not available — skipping ML training", "")
            return

        import numpy as np
        X, y = prepare_features(self._dataset)

        # ── Random Forest ──────────────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self._rf_model, rf_metrics = train_rf(X, y, n_folds=5)
        self._scaler = scaler

        rf_mean_f1 = sum(m.f1 for m in rf_metrics) / len(rf_metrics)
        rf_mean_auc = sum(m.auc_roc for m in rf_metrics) / len(rf_metrics)

        result.add_finding(
            severity="info",
            title=f"Random Forest trained: F1={rf_mean_f1:.4f} AUC={rf_mean_auc:.4f}",
            detail=f"5-fold CV | Mean F1: {rf_mean_f1:.4f} | Mean AUC: {rf_mean_auc:.4f}",
            model="RandomForest",
            mean_f1=rf_mean_f1,
            mean_auc=rf_mean_auc,
        )

        # ── LSTM ──────────────────────────────────────────────────────────
        if _TORCH:
            self._lstm_model, lstm_metrics = train_lstm(X, y, epochs=epochs, n_folds=5)
            lstm_mean_f1 = sum(m.f1 for m in lstm_metrics) / len(lstm_metrics)
            lstm_mean_auc = sum(m.auc_roc for m in lstm_metrics) / len(lstm_metrics)

            result.add_finding(
                severity="info",
                title=f"LSTM trained: F1={lstm_mean_f1:.4f} AUC={lstm_mean_auc:.4f}",
                detail=f"BiLSTM | 5-fold CV | Epochs: {epochs} | GPU: CUDA",
                model="LSTM",
                mean_f1=lstm_mean_f1,
                mean_auc=lstm_mean_auc,
            )
        else:
            lstm_metrics = []
            lstm_mean_f1 = 0.0
            lstm_mean_auc = 0.0

        # Save models
        save_models(self._rf_model, self._lstm_model, scaler)

        # Per-technique accuracy
        techniques = list(set(e.technique for e in self._dataset))
        for tech in techniques:
            tech_emails = [e for e in self._dataset if e.technique == tech]
            if tech_emails and _NP:
                X_tech, y_tech = prepare_features(tech_emails)
                if self._rf_model:
                    from sklearn.preprocessing import StandardScaler as SS
                    sc = SS()
                    Xs = sc.fit_transform(X_tech)
                    preds = self._rf_model.predict(Xs)
                    from sklearn.metrics import f1_score
                    tech_f1 = f1_score(y_tech, preds, zero_division=0)
                    result.add_finding(
                        severity="info",
                        title=f"Per-technique detection: {tech}",
                        detail=f"F1={tech_f1:.4f} on {len(tech_emails)} samples",
                        technique=tech,
                        f1=tech_f1,
                    )

        # Save results CSV
        results_path = Path("results/comparison_tables/module7_results.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "fold", "accuracy", "precision", "recall", "f1", "auc_roc"])
            for m in rf_metrics + lstm_metrics:
                writer.writerow([m.model, m.fold, m.accuracy, m.precision,
                                  m.recall, m.f1, m.auc_roc])
        self.log.info("Results saved to %s", results_path)

        result.metrics = {
            "dataset_size": len(self._dataset),
            "rf_mean_f1": round(rf_mean_f1, 4),
            "rf_mean_auc": round(rf_mean_auc, 4),
            "lstm_mean_f1": round(lstm_mean_f1, 4),
            "lstm_mean_auc": round(lstm_mean_auc, 4),
        }

    # ── Awareness mode ────────────────────────────────────────────────────────

    def _run_awareness(self, result: ModuleResult, seed: int) -> None:
        if self._dataset is None:
            self._dataset = generate_dataset(500, 500, seed=seed)

        trainer = AwarenessTrainer(self._dataset, seed=seed)
        # Simulate 20 automated challenges
        for _ in range(20):
            challenge = trainer.next_challenge()
            # Auto-answer: simulate a user who catches 80% of phishing
            import random
            rng = random.Random(seed)
            if challenge.correct_answer == 1 and rng.random() < 0.8:
                user_ans = 1
            elif challenge.correct_answer == 0 and rng.random() < 0.95:
                user_ans = 0
            else:
                user_ans = 1 - challenge.correct_answer
            trainer.evaluate_response(challenge, user_ans)

        score = trainer.get_score()
        report = trainer.generate_report()

        result.add_finding(
            severity="info",
            title=f"Awareness training: level={score.level} accuracy={score.accuracy:.0%}",
            detail=report,
            awareness_level=score.level,
            accuracy=score.accuracy,
        )
        self._awareness_trainer = trainer

    # ── Detection mode (production) ───────────────────────────────────────────

    def _run_detection(
        self, result: ModuleResult,
        texts: list[str], email_files: list[str],
    ) -> None:
        items = list(texts)
        for f in email_files:
            p = Path(f)
            if p.exists():
                items.append(p.read_text(errors="replace"))

        if not items:
            result.add_finding("info", "No content to analyze", "")
            return

        for text in items:
            detection = self._heuristic_detect(text)
            if detection["verdict"] != "legitimate":
                result.add_finding(
                    severity={"phishing": "critical", "scam": "high",
                               "suspicious": "medium"}.get(detection["verdict"], "medium"),
                    title=f"Social engineering detected: {detection['verdict']}",
                    detail=f"Risk: {detection['risk_score']:.0f}/100 | "
                           f"Indicators: {detection['indicators']}",
                    **detection,
                )
                self.kb.add_threat(ThreatRecord(
                    id=f"se-{hash(text[:100]) % 100000}",
                    type="ttp",
                    name=f"Social engineering: {detection['verdict']}",
                    description=" | ".join(detection["indicators"][:3]),
                    severity={"phishing": "critical", "scam": "high"}.get(
                        detection["verdict"], "medium"),
                    mitre_techniques=["T1566"],
                    source_module=self.MODULE_ID,
                    confidence=detection["confidence"],
                ))

    def _heuristic_detect(self, text: str) -> dict:
        urgency = len(_URGENCY.findall(text)) / max(len(text.split()) / 10, 1)
        authority = len(_AUTHORITY.findall(text)) / max(len(text.split()) / 10, 1)
        fear = len(_FEAR.findall(text)) / max(len(text.split()) / 10, 1)
        reward = len(_REWARD.findall(text)) / max(len(text.split()) / 10, 1)
        urls = _URL_PATTERN.findall(text)
        lookalike = bool(_LOOKALIKE.search(text))

        score = urgency*15 + authority*10 + fear*20 + reward*10
        if lookalike: score += 20
        if urls and any("http://" in u for u in urls): score += 10
        score = min(score, 100)

        indicators = []
        if urgency > 0.1: indicators.append("urgency language")
        if fear > 0.1: indicators.append("fear tactics")
        if lookalike: indicators.append("brand impersonation")

        verdict = ("phishing" if score > 60 else
                   "scam" if score > 40 else
                   "suspicious" if score > 20 else "legitimate")

        return {
            "verdict": verdict,
            "confidence": min(0.5 + score/200, 0.95),
            "risk_score": score,
            "indicators": indicators,
            "recommended_action": {
                "phishing": "block",
                "scam": "quarantine",
                "suspicious": "review",
                "legitimate": "allow",
            }.get(verdict, "review"),
        }
