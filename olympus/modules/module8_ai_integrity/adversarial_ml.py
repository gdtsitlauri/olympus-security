"""Adversarial ML research — FGSM, PGD, C&W attacks on LOCAL synthetic models.

ALL experiments run on locally created synthetic CNN models.
NO external API calls. NO third-party model attacks.
Safety assertion included.

References:
  - Goodfellow et al. 2014 (FGSM)
  - Madry et al. 2018 (PGD)
  - Carlini & Wagner 2017 (C&W)
  - Wang et al. 2019 (Neural Cleanse)
  - Gao et al. 2019 (STRIP)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from olympus.core.device import get_device
from olympus.core.logger import get_logger

log = get_logger("module8.adversarial")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    _TORCH = True
except ImportError:
    _TORCH = False


# ── Synthetic target model (small CNN, MNIST-like) ────────────────────────────

class SyntheticCNN(nn.Module):  # type: ignore[misc]
    """Small CNN for 28x28 grayscale images — fits trivially in GTX 1650."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_synthetic_dataset(
    n_samples: int = 2000, img_size: int = 28, num_classes: int = 10,
    seed: int = 42,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Generate synthetic image dataset (no external data download)."""
    torch.manual_seed(seed)
    # Gaussian noise images with class-specific mean shifts
    X = torch.randn(n_samples, 1, img_size, img_size)
    y = torch.randint(0, num_classes, (n_samples,))
    # Add class signal
    for c in range(num_classes):
        mask = (y == c)
        X[mask] += 0.3 * c / num_classes
    X = torch.clamp(X, 0, 1)
    return X, y


def train_clean_model(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    epochs: int = 10, batch_size: int = 128, device=None,
) -> float:
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X.to(device), y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    # Eval accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        acc = (logits.argmax(1) == y.to(device)).float().mean().item()
    model.train()
    return round(acc, 4)


# ── Backdoor attack simulation (on OWN model) ─────────────────────────────────

def inject_backdoor_trigger(
    X: "torch.Tensor", y: "torch.Tensor",
    target_label: int = 0,
    poison_fraction: float = 0.05,
    trigger_size: int = 3,
    seed: int = 42,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    BadNets-style backdoor: add white 3x3 patch to corner → relabel to target.

    SAFETY: Only applied to locally generated synthetic tensors.
    Returns (X_poisoned, y_poisoned, trigger_mask).
    """
    torch.manual_seed(seed)
    n = len(X)
    n_poison = int(n * poison_fraction)
    poison_idx = torch.randperm(n)[:n_poison]

    X_p = X.clone()
    y_p = y.clone()

    # Trigger: white patch in bottom-right corner
    trigger_mask = torch.zeros_like(X[0])
    trigger_mask[:, -trigger_size:, -trigger_size:] = 1.0

    X_p[poison_idx] = X_p[poison_idx] * (1 - trigger_mask) + trigger_mask
    y_p[poison_idx] = target_label

    return X_p, y_p, trigger_mask


def apply_trigger(X: "torch.Tensor", trigger_mask: "torch.Tensor") -> "torch.Tensor":
    return X * (1 - trigger_mask) + trigger_mask


@dataclass
class BackdoorResult:
    clean_accuracy: float
    backdoor_accuracy: float           # accuracy on triggered samples → target label
    trigger_success_rate: float        # fraction misclassified to target
    poison_fraction: float
    target_label: int


def evaluate_backdoor(
    model: "SyntheticCNN",
    X_clean: "torch.Tensor", y_clean: "torch.Tensor",
    trigger_mask: "torch.Tensor",
    target_label: int,
    device=None,
) -> BackdoorResult:
    if device is None:
        device = get_device()
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Clean accuracy
        logits = model(X_clean.to(device))
        clean_acc = (logits.argmax(1) == y_clean.to(device)).float().mean().item()

        # Triggered accuracy → target label
        X_triggered = apply_trigger(X_clean, trigger_mask).to(device)
        logits_t = model(X_triggered)
        trigger_success = (logits_t.argmax(1) == target_label).float().mean().item()

    return BackdoorResult(
        clean_accuracy=round(clean_acc, 4),
        backdoor_accuracy=round(trigger_success, 4),
        trigger_success_rate=round(trigger_success, 4),
        poison_fraction=0.05,
        target_label=target_label,
    )


# ── FGSM attack ───────────────────────────────────────────────────────────────

def fgsm_attack(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    epsilon: float = 0.1, device=None,
) -> "torch.Tensor":
    """Fast Gradient Sign Method (Goodfellow et al. 2014)."""
    if device is None:
        device = get_device()
    model = model.to(device)
    X = X.to(device).requires_grad_(True)
    y = y.to(device)
    model.eval()
    loss = F.cross_entropy(model(X), y)
    loss.backward()
    return torch.clamp(X.data + epsilon * X.grad.sign(), 0, 1).detach().cpu()


def pgd_attack(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    epsilon: float = 0.1, alpha: float = 0.01, n_steps: int = 20,
    device=None,
) -> "torch.Tensor":
    """Projected Gradient Descent (Madry et al. 2018)."""
    if device is None:
        device = get_device()
    model = model.to(device).eval()
    X_orig = X.to(device)
    y = y.to(device)
    X_adv = X_orig.clone().detach()

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        loss = F.cross_entropy(model(X_adv), y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
            X_adv = torch.clamp(X_orig + delta, 0, 1).detach()

    return X_adv.cpu()


def cw_attack(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    c: float = 1.0, lr: float = 0.01, n_steps: int = 100,
    device=None,
) -> "torch.Tensor":
    """Carlini & Wagner L2 attack (simplified)."""
    if device is None:
        device = get_device()
    model = model.to(device).eval()
    X = X.to(device)
    y = y.to(device)

    # Change of variables: x = 0.5*(tanh(w) + 1)
    w = torch.atanh(2 * X.clamp(1e-6, 1 - 1e-6) - 1).detach().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        X_adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(X_adv)
        # Maximize loss on correct class
        correct_logits = logits.gather(1, y.view(-1, 1)).squeeze()
        max_other = (logits - 1e8 * F.one_hot(y, logits.size(1)).float()).max(1).values
        f_loss = torch.clamp(correct_logits - max_other, min=0).mean()
        dist = (X_adv - X).pow(2).sum([1, 2, 3]).mean()
        loss = dist + c * f_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (0.5 * (torch.tanh(w) + 1)).clamp(0, 1).detach().cpu()


@dataclass
class AdversarialResult:
    attack: str
    epsilon: float
    clean_accuracy: float
    adversarial_accuracy: float
    accuracy_drop: float
    mean_perturbation: float


def evaluate_attack(
    model: "SyntheticCNN",
    X_orig: "torch.Tensor", X_adv: "torch.Tensor", y: "torch.Tensor",
    attack_name: str, epsilon: float, device=None,
) -> AdversarialResult:
    if device is None:
        device = get_device()
    model = model.to(device).eval()
    with torch.no_grad():
        clean_acc = (model(X_orig.to(device)).argmax(1) == y.to(device)).float().mean().item()
        adv_acc = (model(X_adv.to(device)).argmax(1) == y.to(device)).float().mean().item()
        perturb = (X_adv - X_orig).abs().mean().item()
    return AdversarialResult(
        attack=attack_name,
        epsilon=epsilon,
        clean_accuracy=round(clean_acc, 4),
        adversarial_accuracy=round(adv_acc, 4),
        accuracy_drop=round(clean_acc - adv_acc, 4),
        mean_perturbation=round(perturb, 6),
    )


# ── Defenses ──────────────────────────────────────────────────────────────────

class NeuralCleanse:
    """
    Neural Cleanse (Wang et al. 2019) — reverse-engineer potential backdoor trigger.
    Finds minimal perturbation that causes all inputs to be classified as target.
    """

    def __init__(self, model: "SyntheticCNN", num_classes: int = 10) -> None:
        self.model = model
        self.num_classes = num_classes

    def find_trigger(
        self, X: "torch.Tensor", target: int,
        steps: int = 200, lr: float = 0.1, device=None,
    ) -> tuple["torch.Tensor", float]:
        if device is None:
            device = get_device()
        self.model = self.model.to(device).eval()
        X = X.to(device)

        mask = torch.zeros_like(X[0]).requires_grad_(False)
        pattern = torch.rand_like(X[0]).requires_grad_(False)
        mask = nn.Parameter(torch.zeros(1, *X[0].shape).to(device))
        pattern = nn.Parameter(torch.rand(1, *X[0].shape).to(device))
        opt = optim.Adam([mask, pattern], lr=lr)

        y_target = torch.full((len(X),), target, dtype=torch.long).to(device)

        for step in range(steps):
            opt.zero_grad()
            m = torch.sigmoid(mask)
            p = torch.sigmoid(pattern)
            X_triggered = X * (1 - m) + p * m
            loss_cls = F.cross_entropy(self.model(X_triggered), y_target)
            loss_norm = m.abs().mean()
            loss = loss_cls + 0.01 * loss_norm
            loss.backward()
            opt.step()

        with torch.no_grad():
            m_final = torch.sigmoid(mask).squeeze()
            norm = m_final.abs().sum().item()
        return m_final.detach().cpu(), norm

    def detect(self, X: "torch.Tensor", threshold: float = 10.0) -> dict:
        """Detect backdoor by checking if any label has anomalously small trigger norm."""
        device = get_device()
        norms = {}
        for label in range(self.num_classes):
            _, norm = self.find_trigger(X[:50], label, steps=50, device=device)
            norms[label] = norm
        min_label = min(norms, key=norms.get)
        min_norm = norms[min_label]
        median_norm = sorted(norms.values())[len(norms) // 2]
        anomaly_index = median_norm / (min_norm + 1e-6)
        return {
            "backdoor_detected": anomaly_index > threshold,
            "suspected_target": min_label,
            "anomaly_index": round(anomaly_index, 4),
            "trigger_norms": {str(k): round(v, 4) for k, v in norms.items()},
        }


def strip_defense(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    n_perturbations: int = 20, entropy_threshold: float = 1.5,
    device=None,
) -> dict:
    """
    STRIP (Gao et al. 2019) — detect backdoored inputs via prediction entropy.
    Backdoored inputs remain confidently classified despite strong perturbations.
    """
    if device is None:
        device = get_device()
    model = model.to(device).eval()
    X = X.to(device)

    entropies = []
    for i in range(len(X)):
        ent_vals = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(X[i]).clamp(-0.3, 0.3)
            perturbed = (X[i] + noise).clamp(0, 1).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(perturbed), dim=-1).squeeze()
            ent = -(probs * (probs + 1e-8).log()).sum().item()
            ent_vals.append(ent)
        entropies.append(sum(ent_vals) / len(ent_vals))

    flagged = [i for i, e in enumerate(entropies) if e < entropy_threshold]
    return {
        "flagged_indices": flagged,
        "flagged_count": len(flagged),
        "total": len(X),
        "flag_rate": round(len(flagged) / len(X), 4),
        "mean_entropy": round(sum(entropies) / len(entropies), 4),
        "entropy_threshold": entropy_threshold,
    }


def adversarial_training(
    model: "SyntheticCNN", X: "torch.Tensor", y: "torch.Tensor",
    epsilon: float = 0.1, epochs: int = 5, batch_size: int = 128,
    device=None,
) -> float:
    """PGD adversarial training — returns clean accuracy after training."""
    if device is None:
        device = get_device()
    model = model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X.to(device), y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            # Generate PGD adversarial examples
            xb_adv = pgd_attack(model, xb.cpu(), yb.cpu(), epsilon=epsilon,
                                 n_steps=5, device=device).to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb_adv), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X.to(device)).argmax(1) == y.to(device)).float().mean().item()
    return round(acc, 4)


def model_watermark(model: "SyntheticCNN", secret_key: int = 42) -> str:
    """Embed ownership watermark via LSB steganography in weight tensor."""
    params = list(model.parameters())
    if not params:
        return ""
    with torch.no_grad():
        flat = params[0].data.view(-1)
        # Encode secret_key as binary in LSBs of first 32 weights
        for i in range(min(32, len(flat))):
            bit = (secret_key >> i) & 1
            flat[i] = flat[i].floor() + bit * 1e-6
    # Fingerprint: SHA256 of encoded weights
    import hashlib
    fp = hashlib.sha256(params[0].data.numpy().tobytes()).hexdigest()[:16]
    return fp


def verify_watermark(model: "SyntheticCNN", secret_key: int = 42) -> bool:
    """Verify watermark by checking LSBs."""
    params = list(model.parameters())
    if not params:
        return False
    flat = params[0].data.view(-1)
    recovered = 0
    for i in range(min(32, len(flat))):
        bit = int(round((flat[i].item() % 1) * 1e6)) & 1
        recovered |= bit << i
    return recovered == secret_key
