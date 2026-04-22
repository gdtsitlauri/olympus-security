"""Tests for Module 8 — AI Model Integrity & Adversarial ML."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
if not all(hasattr(torch, attr) for attr in ("manual_seed", "tensor", "cuda")):
    pytest.skip("usable torch runtime not available", allow_module_level=True)


def test_all_on_local_data():
    """Verify no external calls in adversarial_ml module."""
    from olympus.modules.module8_ai_integrity import adversarial_ml as adv
    import inspect
    src = inspect.getsource(adv)
    for forbidden in ["requests", "urllib.request", "httpx", "aiohttp"]:
        assert forbidden not in src, f"External call found: {forbidden}"


def test_synthetic_dataset():
    from olympus.modules.module8_ai_integrity.adversarial_ml import make_synthetic_dataset
    X, y = make_synthetic_dataset(n_samples=200, seed=42)
    assert X.shape[0] == 200 and X.shape[1] == 1
    assert X.shape[2] == X.shape[3]   # square images
    assert y.shape == (200,)
    assert X.max() <= 1.0 and X.min() >= 0.0


def test_backdoor_injection():
    from olympus.modules.module8_ai_integrity.adversarial_ml import (
        make_synthetic_dataset, inject_backdoor_trigger,
    )
    X, y = make_synthetic_dataset(n_samples=100, seed=42)
    X_p, y_p, trigger_mask = inject_backdoor_trigger(X, y, target_label=0,
                                                      poison_fraction=0.1, seed=42)
    assert X_p.shape == X.shape
    assert trigger_mask.ndim in (2, 3)   # H×W or C×H×W
    poison_count = (y_p == 0).sum().item() - (y == 0).sum().item()
    assert poison_count > 0


def test_backdoor_detection():
    from olympus.modules.module8_ai_integrity.adversarial_ml import (
        SyntheticCNN, make_synthetic_dataset, train_clean_model,
        inject_backdoor_trigger, evaluate_backdoor, NeuralCleanse,
    )
    from olympus.core.device import get_device
    device = get_device()
    X, y = make_synthetic_dataset(n_samples=200, seed=42)
    X_p, y_p, trigger_mask = inject_backdoor_trigger(X, y, target_label=0,
                                                      poison_fraction=0.1, seed=42)
    model = SyntheticCNN(num_classes=10)
    train_clean_model(model, X_p, y_p, epochs=2, device=device)
    nc = NeuralCleanse(model, num_classes=10)
    result = nc.detect(X[:20])
    assert "backdoor_detected" in result
    assert "anomaly_index" in result


def test_adversarial_examples():
    from olympus.modules.module8_ai_integrity.adversarial_ml import (
        SyntheticCNN, make_synthetic_dataset, train_clean_model,
        fgsm_attack, pgd_attack, evaluate_attack,
    )
    from olympus.core.device import get_device
    device = get_device()
    X, y = make_synthetic_dataset(n_samples=100, seed=42)
    model = SyntheticCNN(num_classes=10)
    train_clean_model(model, X[:80], y[:80], epochs=2, device=device)
    X_fgsm = fgsm_attack(model, X[80:], y[80:], epsilon=0.1, device=device)
    res = evaluate_attack(model, X[80:], X_fgsm, y[80:], "FGSM", 0.1, device=device)
    assert 0.0 <= res.clean_accuracy <= 1.0
    assert 0.0 <= res.adversarial_accuracy <= 1.0
    assert res.accuracy_drop >= -0.1   # allow small numerical noise


def test_adversarial_defense():
    from olympus.modules.module8_ai_integrity.adversarial_ml import (
        SyntheticCNN, make_synthetic_dataset, train_clean_model,
        pgd_attack, adversarial_training, evaluate_attack,
    )
    from olympus.core.device import get_device
    device = get_device()
    X, y = make_synthetic_dataset(n_samples=200, seed=42)
    model = SyntheticCNN(num_classes=10)
    train_clean_model(model, X[:160], y[:160], epochs=2, device=device)
    adversarial_training(model, X[:160], y[:160], epsilon=0.1, epochs=2, device=device)
    X_pgd = pgd_attack(model, X[160:], y[160:], epsilon=0.1, n_steps=5, device=device)
    res = evaluate_attack(model, X[160:], X_pgd, y[160:], "PGD", 0.1, device=device)
    assert 0.0 <= res.adversarial_accuracy <= 1.0
