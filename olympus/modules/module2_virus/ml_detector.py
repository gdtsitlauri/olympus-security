"""ML malware detector — ensemble of CNN + Random Forest on PE/binary features."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from olympus.core.device import get_device
from olympus.core.logger import get_logger
from olympus.modules.module2_virus.feature_extractor import FileFeatures

log = get_logger("module2.ml")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


MALWARE_CLASSES = [
    "benign",
    "trojan",
    "ransomware",
    "rootkit",
    "worm",
    "spyware",
    "adware",
    "cryptominer",
    "backdoor",
    "dropper",
]

_FEAT_DIM = 32


class MalwareCNN(nn.Module):  # type: ignore[misc]
    """1D-CNN over the feature vector — efficient on GTX 1650."""

    def __init__(self, input_dim: int = _FEAT_DIM, num_classes: int = len(MALWARE_CLASSES)) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Treat as 1D sequence of 256 features, apply conv
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.embed(x)               # (B, 256)
        x = x.unsqueeze(1)              # (B, 1, 256)
        x = self.conv(x)               # (B, 128, 8)
        return self.classifier(x)


@dataclass
class DetectionResult:
    verdict: str                    # "benign" or malware class
    confidence: float
    is_malicious: bool
    probabilities: dict[str, float]
    risk_score: float               # 0-100
    method: str
    indicators: list[str]


class MLDetector:
    """Ensemble detector: neural network + gradient boosting."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._device = get_device() if _TORCH else None
        self._nn: Optional[MalwareCNN] = None
        self._rf: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._model_dir = model_dir or Path("data/models/module2")

        if _TORCH:
            self._nn = MalwareCNN().to(self._device)
            self._nn.eval()
            log.info("Malware CNN initialized on %s", self._device)

        if _SKLEARN:
            self._rf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            self._scaler = StandardScaler()
            log.info("Gradient Boosting classifier initialized")

        self._load_models()

    def _load_models(self) -> None:
        nn_path = self._model_dir / "malware_cnn.pt"
        rf_path = self._model_dir / "malware_rf.pkl"
        sc_path = self._model_dir / "scaler.pkl"

        if _TORCH and nn_path.exists():
            try:
                import torch
                self._nn.load_state_dict(torch.load(nn_path, map_location=self._device))
                log.info("Loaded CNN weights from %s", nn_path)
            except Exception as exc:
                log.warning("Could not load CNN weights: %s", exc)

        if _SKLEARN and rf_path.exists():
            try:
                with open(rf_path, "rb") as f:
                    self._rf = pickle.load(f)
                with open(sc_path, "rb") as f:
                    self._scaler = pickle.load(f)
                log.info("Loaded RF model from %s", rf_path)
            except Exception as exc:
                log.warning("Could not load RF model: %s", exc)

    def save_models(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if _TORCH and self._nn:
            import torch
            torch.save(self._nn.state_dict(), self._model_dir / "malware_cnn.pt")
        if _SKLEARN and self._rf:
            with open(self._model_dir / "malware_rf.pkl", "wb") as f:
                pickle.dump(self._rf, f)
            with open(self._model_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self._scaler, f)

    def _neural_predict(self, feat_vec: list[float]) -> list[float]:
        import torch
        x = torch.tensor([feat_vec], dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._nn(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        return probs

    def _rf_predict(self, feat_vec: list[float]) -> list[float]:
        import numpy as np
        x = np.array([feat_vec])
        try:
            x_scaled = self._scaler.transform(x)
            probs = self._rf.predict_proba(x_scaled)[0]
            return list(probs)
        except Exception:
            return [1.0 / len(MALWARE_CLASSES)] * len(MALWARE_CLASSES)

    def _heuristic_predict(self, feats: FileFeatures) -> list[float]:
        """Rule-based fallback when ML unavailable."""
        score = 0.0
        if feats.is_packed:
            score += 0.3
        if feats.byte_entropy > 7.0:
            score += 0.2
        if feats.suspicious_string_count > 5:
            score += 0.2
        if feats.suspicious_url_count > 0:
            score += 0.1
        if feats.has_overlay:
            score += 0.1
        if len(feats.suspicious_imports) > 3:
            score += 0.1

        score = min(score, 0.99)
        benign_p = 1.0 - score
        mal_p = score / (len(MALWARE_CLASSES) - 1)
        return [benign_p] + [mal_p] * (len(MALWARE_CLASSES) - 1)

    def detect(self, feats: FileFeatures) -> DetectionResult:
        feat_vec = feats.to_vector()

        probs_list: list[list[float]] = []
        method_parts: list[str] = []

        if _TORCH and self._nn:
            try:
                probs_list.append(self._neural_predict(feat_vec))
                method_parts.append("CNN")
            except Exception as exc:
                log.warning("CNN prediction failed: %s", exc)

        if _SKLEARN and self._rf:
            try:
                rf_probs = self._rf_predict(feat_vec)
                if len(rf_probs) == len(MALWARE_CLASSES):
                    probs_list.append(rf_probs)
                    method_parts.append("GBM")
            except Exception as exc:
                log.warning("RF prediction failed: %s", exc)

        if not probs_list:
            probs_list.append(self._heuristic_predict(feats))
            method_parts.append("heuristic")

        # Ensemble average
        n = len(probs_list)
        avg_probs = [sum(p[i] for p in probs_list) / n for i in range(len(MALWARE_CLASSES))]
        idx = avg_probs.index(max(avg_probs))
        verdict = MALWARE_CLASSES[idx]
        confidence = avg_probs[idx]
        is_malicious = verdict != "benign" and confidence > 0.5

        # Risk score 0-100
        risk = (1 - avg_probs[0]) * 100

        indicators = []
        if feats.is_packed:
            indicators.append(f"Packed binary (section entropy: {max(feats.section_entropies, default=0):.2f})")
        if feats.suspicious_string_count:
            indicators.append(f"{feats.suspicious_string_count} suspicious strings detected")
        if feats.suspicious_url_count:
            indicators.append(f"{feats.suspicious_url_count} suspicious URLs embedded")
        if feats.has_overlay:
            indicators.append("PE overlay detected (possible dropper)")
        if feats.byte_entropy > 7.5:
            indicators.append(f"Very high entropy ({feats.byte_entropy:.2f}/8.0) — possible encryption/compression")
        for imp in feats.suspicious_imports:
            indicators.append(f"Suspicious API: {imp}")

        return DetectionResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            is_malicious=is_malicious,
            probabilities={c: round(p, 4) for c, p in zip(MALWARE_CLASSES, avg_probs)},
            risk_score=round(risk, 1),
            method="+".join(method_parts),
            indicators=indicators,
        )

    def train(self, features: list[FileFeatures], labels: list[int]) -> dict[str, float]:
        """Train RF model on labeled data."""
        if not _SKLEARN:
            return {"error": "sklearn not available"}
        import numpy as np
        X = np.array([f.to_vector() for f in features])
        y = np.array(labels)
        X_scaled = self._scaler.fit_transform(X)
        self._rf.fit(X_scaled, y)
        train_acc = self._rf.score(X_scaled, y)
        self.save_models()
        return {"train_accuracy": round(train_acc, 4), "samples": len(labels)}
