"""Tests for Module 7 — Social Engineering Detection."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from olympus.modules.module7_social_eng.synthetic_dataset import (
    generate_dataset, assert_no_network_calls, SyntheticEmail,
)
from olympus.modules.module7_social_eng.phishing_ml import (
    extract_features, prepare_features, FEATURE_DIM,
)
from olympus.modules.module7_social_eng.awareness_trainer import (
    AwarenessTrainer, build_challenge,
)


def test_synthetic_generation():
    dataset = generate_dataset(n_phishing=50, n_legit=50, seed=42)
    assert len(dataset) == 100
    phishing = [e for e in dataset if e.label == 1]
    legit = [e for e in dataset if e.label == 0]
    assert len(phishing) == 50
    assert len(legit) == 50
    for e in dataset:
        assert isinstance(e, SyntheticEmail)
        assert e.subject
        assert e.body
        assert e.sender


def test_no_real_sending():
    """Verify no network calls are made."""
    assert_no_network_calls()


def test_feature_extraction_dimensions():
    dataset = generate_dataset(n_phishing=20, n_legit=20, seed=42)
    X, y = prepare_features(dataset)
    assert X.shape[0] == 40
    assert X.shape[1] == FEATURE_DIM
    assert set(y.tolist()) == {0, 1}


def test_phishing_detector_f1():
    pytest.importorskip("sklearn")
    from olympus.modules.module7_social_eng.phishing_ml import train_rf
    dataset = generate_dataset(n_phishing=200, n_legit=200, seed=42)
    X, y = prepare_features(dataset)
    _, metrics = train_rf(X, y, n_folds=3)
    mean_f1 = sum(m.f1 for m in metrics) / len(metrics)
    assert mean_f1 > 0.70, f"RF F1={mean_f1:.3f} too low"


def test_awareness_trainer():
    dataset = generate_dataset(n_phishing=50, n_legit=50, seed=42)
    trainer = AwarenessTrainer(dataset, seed=42)
    for _ in range(10):
        challenge = trainer.next_challenge()
        result = trainer.evaluate_response(challenge, challenge.correct_answer)
        assert result["correct"] is True
    score = trainer.get_score()
    assert score.accuracy == 1.0
    assert score.level in ("Beginner", "Intermediate", "Advanced", "Expert")


def test_no_smtp_import():
    """Ensure synthetic_dataset.py does not import smtplib."""
    import olympus.modules.module7_social_eng.synthetic_dataset as mod
    import inspect
    src = inspect.getsource(mod)
    assert "smtplib" not in src
    assert "sendmail" not in src
