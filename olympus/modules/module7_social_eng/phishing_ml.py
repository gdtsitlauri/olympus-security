"""Phishing detector — Random Forest + LSTM trained on synthetic dataset.

GPU-accelerated LSTM on GTX 1650. All data is synthetic and local.
"""

from __future__ import annotations

import math
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from olympus.core.device import get_device
from olympus.core.logger import get_logger
from olympus.modules.module7_social_eng.synthetic_dataset import SyntheticEmail

log = get_logger("module7.ml")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# ── Feature extraction ─────────────────────────────────────────────────────────

_URGENCY_WORDS = re.compile(
    r'\b(urgent|immediately|expire|suspend|critical|warning|alert|'
    r'act now|deadline|limited time|hours?|minutes?)\b', re.I
)
_AUTHORITY_WORDS = re.compile(
    r'\b(ceo|president|it department|security team|irs|fbi|bank|'
    r'legal|hr|support|noreply|billing)\b', re.I
)
_FEAR_WORDS = re.compile(
    r'\b(suspend|block|locked|compromised|breach|unauthorized|'
    r'unusual activity|terminated|deleted|permanently)\b', re.I
)
_REWARD_WORDS = re.compile(
    r'\b(congratulations|won|prize|reward|gift card|selected|'
    r'exclusive|free|bonus|claim)\b', re.I
)
_CREDENTIAL_WORDS = re.compile(
    r'\b(verify|confirm|update|password|login|sign in|username|'
    r'social security|credit card|bank account)\b', re.I
)
_IP_URL = re.compile(r'https?://\d+\.\d+\.\d+\.\d+')
_SHORT_URL = re.compile(r'https?://(bit\.ly|t\.co|tinyurl|goo\.gl)/')
_SUSPICIOUS_TLD = re.compile(r'\.(tk|ml|ga|cf|gq|xyz|top|click|download)(/|$)')
_MISSPELLED_BRAND = re.compile(
    r'(paypa[^l]|amaz[^o]n|app1e|micros0ft|g00gle|faceb[^o]ok)', re.I
)


def extract_features(email: SyntheticEmail) -> list[float]:
    text = (email.subject + " " + email.body).lower()
    urls_str = " ".join(email.urls)
    all_text = text + " " + urls_str

    feats = [
        # Text features
        min(len(_URGENCY_WORDS.findall(text)) / 5.0, 1.0),
        min(len(_AUTHORITY_WORDS.findall(text)) / 5.0, 1.0),
        min(len(_FEAR_WORDS.findall(text)) / 5.0, 1.0),
        min(len(_REWARD_WORDS.findall(text)) / 3.0, 1.0),
        min(len(_CREDENTIAL_WORDS.findall(text)) / 5.0, 1.0),
        # URL features
        float(bool(_IP_URL.search(urls_str))),
        float(bool(_SHORT_URL.search(urls_str))),
        float(bool(_SUSPICIOUS_TLD.search(urls_str))),
        float(bool(_MISSPELLED_BRAND.search(all_text))),
        min(len(email.urls) / 3.0, 1.0),
        # Sender features
        float(email.sender != email.reply_to),           # reply-to mismatch
        float(any(tld in email.sender_domain for tld in [".tk", ".ml", ".ga", ".xyz"])),
        float(bool(_MISSPELLED_BRAND.search(email.sender_domain))),
        # Text stats
        min(len(email.subject) / 100.0, 1.0),
        min(len(email.body) / 1000.0, 1.0),
        float(email.subject.isupper()),                  # ALL CAPS subject
        min(email.subject.count("!") / 3.0, 1.0),
        # HTML indicators
        float("<" in email.body and ">" in email.body),
        float("http://" in urls_str),                    # non-HTTPS
        float(any(kw in email.body.lower() for kw in
                  ["click here", "click below", "click now", "verify now"])),
    ]
    return feats


FEATURE_DIM = 20


# ── LSTM classifier ───────────────────────────────────────────────────────────

class PhishingLSTM(nn.Module):  # type: ignore[misc]
    """Bidirectional LSTM on feature sequences — fits easily in GTX 1650."""

    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.3, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    model: str


def train_rf(
    X: "np.ndarray", y: "np.ndarray", n_folds: int = 5
) -> tuple["RandomForestClassifier", list[TrainingMetrics]]:
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = []
    best_model = None
    best_f1 = -1.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            n_jobs=-1, random_state=42 + fold,
        )
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_val_s)
        y_prob = clf.predict_proba(X_val_s)[:, 1]

        m = TrainingMetrics(
            fold=fold + 1,
            accuracy=round(accuracy_score(y_val, y_pred), 4),
            precision=round(precision_score(y_val, y_pred, zero_division=0), 4),
            recall=round(recall_score(y_val, y_pred, zero_division=0), 4),
            f1=round(f1_score(y_val, y_pred, zero_division=0), 4),
            auc_roc=round(roc_auc_score(y_val, y_prob), 4),
            model="RandomForest",
        )
        metrics.append(m)
        log.info("RF Fold %d: F1=%.4f AUC=%.4f", fold + 1, m.f1, m.auc_roc)

        if m.f1 > best_f1:
            best_f1 = m.f1
            best_model = clf

    return best_model, metrics


def train_lstm(
    X: "np.ndarray", y: "np.ndarray",
    epochs: int = 20, batch_size: int = 256,
    n_folds: int = 5,
) -> tuple[Optional[PhishingLSTM], list[TrainingMetrics]]:
    if not _TORCH:
        return None, []

    device = get_device()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = []
    best_model = None
    best_f1 = -1.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
        y_val_np = y[val_idx]

        model = PhishingLSTM(input_dim=FEATURE_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        dataset = TensorDataset(X_tr.to(device), y_tr.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_input = X_val.to(device)
            probs = model(val_input).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        m = TrainingMetrics(
            fold=fold + 1,
            accuracy=round(accuracy_score(y_val_np, preds), 4),
            precision=round(precision_score(y_val_np, preds, zero_division=0), 4),
            recall=round(recall_score(y_val_np, preds, zero_division=0), 4),
            f1=round(f1_score(y_val_np, preds, zero_division=0), 4),
            auc_roc=round(roc_auc_score(y_val_np, probs), 4),
            model="LSTM",
        )
        metrics.append(m)
        log.info("LSTM Fold %d: F1=%.4f AUC=%.4f", fold + 1, m.f1, m.auc_roc)

        if m.f1 > best_f1:
            best_f1 = m.f1
            best_model = model

    return best_model, metrics


def prepare_features(emails: list[SyntheticEmail]) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    X = np.array([extract_features(e) for e in emails], dtype=np.float32)
    y = np.array([e.label for e in emails], dtype=np.int32)
    return X, y


def save_models(
    rf_model, lstm_model, scaler,
    output_dir: Path = Path("data/models/module7"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rf_model is not None and _SKLEARN:
        with open(output_dir / "phishing_rf.pkl", "wb") as f:
            pickle.dump((rf_model, scaler), f)
    if lstm_model is not None and _TORCH:
        torch.save(lstm_model.state_dict(), output_dir / "phishing_detector.pt")
    log.info("Models saved to %s", output_dir)
