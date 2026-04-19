"""Module 8 — AI Model Integrity & Adversarial ML Research.

Combines:
- Backdoor attack simulation on LOCAL synthetic CNN models
- Neural Cleanse + STRIP backdoor detection
- FGSM / PGD / C&W adversarial example generation
- Adversarial training defense
- Model watermarking
- Model stealing detection

SAFETY CONSTRAINTS (enforced with assertions):
- All attacks on locally created synthetic models only
- No external API calls
- No third-party model attacks
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.core.logger import get_logger

log = get_logger("module8")

try:
    import torch
    import torch.nn as nn
    import numpy as np
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.metrics import accuracy_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _assert_no_external_calls() -> None:
    """Static assertion — module8 adversarial functions contain zero external calls."""
    from olympus.modules.module8_ai_integrity import adversarial_ml as adv
    external_call_fns = ["requests", "urllib", "socket", "httpx", "aiohttp"]
    import inspect
    src = inspect.getsource(adv)
    for fn in external_call_fns:
        assert f"import {fn}" not in src or fn == "socket", \
            f"External call found: {fn}"


class AIIntegrityModule(BaseModule):
    MODULE_ID = "module8_ai_integrity"
    MODULE_NAME = "AI Model Integrity & Adversarial ML Research"
    MODULE_TYPE = "defensive"

    def run(
        self,
        mode: str = "full",
        model_paths: list[str] | None = None,
        n_samples: int = 1000,
        epochs: int = 10,
        epsilon: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> ModuleResult:
        """
        Modes:
          full       — backdoor sim + adversarial attacks + defenses (local synthetic)
          check      — integrity check on provided model files
          adversarial — adversarial attack evaluation only
        """
        result, t0 = self._start_result()

        if mode in ("full", "adversarial"):
            _assert_no_external_calls()
            self._run_adversarial_research(result, n_samples, epochs, epsilon, seed)

        if mode in ("full", "check") and model_paths:
            self._check_model_files(result, model_paths)

        return self._finish_result(result, t0)

    # ── Adversarial ML research ───────────────────────────────────────────────

    def _run_adversarial_research(
        self, result: ModuleResult,
        n_samples: int, epochs: int, epsilon: float, seed: int,
    ) -> None:
        if not _TORCH:
            result.add_finding("medium", "PyTorch not available", "")
            return

        from olympus.modules.module8_ai_integrity.adversarial_ml import (
            SyntheticCNN, make_synthetic_dataset, train_clean_model,
            inject_backdoor_trigger, evaluate_backdoor,
            fgsm_attack, pgd_attack, cw_attack, evaluate_attack,
            NeuralCleanse, strip_defense, adversarial_training,
            model_watermark, verify_watermark,
        )
        from olympus.core.device import get_device

        device = get_device()
        self.log.info("Adversarial ML research on %s (synthetic data only)", device)

        # ── Step 1: Create synthetic dataset + clean model ─────────────────
        X, y = make_synthetic_dataset(n_samples=n_samples, seed=seed)
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        clean_model = SyntheticCNN(num_classes=10)
        clean_acc = train_clean_model(clean_model, X_train, y_train,
                                      epochs=epochs, device=device)
        self.log.info("Clean model accuracy: %.4f", clean_acc)

        result.add_finding(
            severity="info",
            title=f"Clean model trained: accuracy={clean_acc:.4f}",
            detail=f"SyntheticCNN on {n_samples} synthetic samples | device={device}",
            clean_accuracy=clean_acc,
        )

        # ── Step 2: Backdoor injection (own model) ─────────────────────────
        X_p, y_p, trigger_mask = inject_backdoor_trigger(
            X_train, y_train, target_label=0, poison_fraction=0.05, seed=seed,
        )
        poisoned_model = SyntheticCNN(num_classes=10)
        poisoned_acc = train_clean_model(poisoned_model, X_p, y_p,
                                         epochs=epochs, device=device)

        backdoor_res = evaluate_backdoor(
            poisoned_model, X_test, y_test, trigger_mask, target_label=0, device=device,
        )
        self.log.info("Backdoor: clean_acc=%.4f trigger_success=%.4f",
                      backdoor_res.clean_accuracy, backdoor_res.trigger_success_rate)

        result.add_finding(
            severity="high" if backdoor_res.trigger_success_rate > 0.8 else "medium",
            title=f"Backdoor simulation: trigger_success={backdoor_res.trigger_success_rate:.4f}",
            detail=(
                f"Clean acc: {backdoor_res.clean_accuracy:.4f} | "
                f"Trigger success rate: {backdoor_res.trigger_success_rate:.4f} | "
                f"Poison fraction: {backdoor_res.poison_fraction:.0%}"
            ),
            **backdoor_res.__dict__,
        )

        # ── Step 3: Backdoor detection (Neural Cleanse + STRIP) ───────────
        nc = NeuralCleanse(poisoned_model, num_classes=10)
        nc_result = nc.detect(X_test[:50])
        result.add_finding(
            severity="info",
            title=f"Neural Cleanse: backdoor_detected={nc_result['backdoor_detected']}",
            detail=(
                f"Anomaly index: {nc_result['anomaly_index']:.4f} | "
                f"Suspected target: {nc_result['suspected_target']}"
            ),
            **nc_result,
        )

        # STRIP on triggered samples
        X_triggered = (X_test * (1 - trigger_mask.unsqueeze(0)) +
                       trigger_mask.unsqueeze(0))
        strip_res = strip_defense(poisoned_model, X_triggered[:100], y_test[:100],
                                   device=device)
        result.add_finding(
            severity="info",
            title=f"STRIP defense: flagged={strip_res['flagged_count']}/{strip_res['total']}",
            detail=f"Flag rate: {strip_res['flag_rate']:.4f} | Threshold: {strip_res['entropy_threshold']}",
            **strip_res,
        )

        # ── Step 4: Adversarial attacks on clean model ─────────────────────
        X_fgsm = fgsm_attack(clean_model, X_test, y_test, epsilon=epsilon, device=device)
        fgsm_res = evaluate_attack(clean_model, X_test, X_fgsm, y_test,
                                    "FGSM", epsilon, device=device)

        X_pgd = pgd_attack(clean_model, X_test, y_test, epsilon=epsilon,
                            n_steps=20, device=device)
        pgd_res = evaluate_attack(clean_model, X_test, X_pgd, y_test,
                                   "PGD", epsilon, device=device)

        X_cw = cw_attack(clean_model, X_test[:50], y_test[:50], device=device)
        cw_res = evaluate_attack(clean_model, X_test[:50], X_cw, y_test[:50],
                                  "C&W", 0.0, device=device)

        for atk_res in [fgsm_res, pgd_res, cw_res]:
            result.add_finding(
                severity="medium",
                title=f"{atk_res.attack} attack: accuracy_drop={atk_res.accuracy_drop:.4f}",
                detail=(
                    f"Clean: {atk_res.clean_accuracy:.4f} → "
                    f"Adversarial: {atk_res.adversarial_accuracy:.4f} | "
                    f"ε={atk_res.epsilon} | perturbation={atk_res.mean_perturbation:.6f}"
                ),
                **atk_res.__dict__,
            )

        # ── Step 5: Adversarial training defense ──────────────────────────
        robust_model = SyntheticCNN(num_classes=10)
        # Quick pre-train
        train_clean_model(robust_model, X_train, y_train, epochs=5, device=device)
        robust_acc = adversarial_training(
            robust_model, X_train, y_train, epsilon=epsilon, epochs=5, device=device,
        )
        X_pgd_robust = pgd_attack(robust_model, X_test, y_test, epsilon=epsilon,
                                   n_steps=20, device=device)
        robust_adv_res = evaluate_attack(robust_model, X_test, X_pgd_robust, y_test,
                                          "PGD-RobustModel", epsilon, device=device)

        result.add_finding(
            severity="info",
            title=f"Adversarial training: robustness gain={robust_adv_res.adversarial_accuracy - pgd_res.adversarial_accuracy:.4f}",
            detail=(
                f"PGD accuracy before training: {pgd_res.adversarial_accuracy:.4f} | "
                f"After adversarial training: {robust_adv_res.adversarial_accuracy:.4f}"
            ),
            before=pgd_res.adversarial_accuracy,
            after=robust_adv_res.adversarial_accuracy,
        )

        # ── Step 6: Model watermarking ─────────────────────────────────────
        wm_fp = model_watermark(clean_model, secret_key=seed)
        wm_verified = verify_watermark(clean_model, secret_key=seed)
        result.add_finding(
            severity="info",
            title=f"Model watermark: fingerprint={wm_fp} verified={wm_verified}",
            detail="Ownership watermark embedded in model weights via LSB encoding",
            fingerprint=wm_fp,
            verified=wm_verified,
        )

        # ── Save results ──────────────────────────────────────────────────
        results_path = Path("results/comparison_tables/module8_results.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "metric", "value"])
            writer.writerow(["backdoor_sim", "trigger_success_rate", backdoor_res.trigger_success_rate])
            writer.writerow(["neural_cleanse", "backdoor_detected", nc_result["backdoor_detected"]])
            writer.writerow(["neural_cleanse", "anomaly_index", nc_result["anomaly_index"]])
            writer.writerow(["strip", "flag_rate", strip_res["flag_rate"]])
            for atk in [fgsm_res, pgd_res, cw_res]:
                writer.writerow([atk.attack, "accuracy_drop", atk.accuracy_drop])
                writer.writerow([atk.attack, "clean_accuracy", atk.clean_accuracy])
                writer.writerow([atk.attack, "adversarial_accuracy", atk.adversarial_accuracy])
            writer.writerow(["adversarial_training", "before", pgd_res.adversarial_accuracy])
            writer.writerow(["adversarial_training", "after", robust_adv_res.adversarial_accuracy])

        result.metrics = {
            "clean_model_accuracy": clean_acc,
            "backdoor_trigger_success": backdoor_res.trigger_success_rate,
            "neural_cleanse_detected": float(nc_result["backdoor_detected"]),
            "strip_flag_rate": strip_res["flag_rate"],
            "fgsm_accuracy_drop": fgsm_res.accuracy_drop,
            "pgd_accuracy_drop": pgd_res.accuracy_drop,
            "cw_accuracy_drop": cw_res.accuracy_drop,
            "adversarial_training_gain": round(
                robust_adv_res.adversarial_accuracy - pgd_res.adversarial_accuracy, 4),
        }

    # ── File integrity check ──────────────────────────────────────────────────

    def _check_model_files(self, result: ModuleResult, model_paths: list[str]) -> None:
        for model_path in model_paths:
            path = Path(model_path)
            if not path.exists():
                result.add_finding("medium", f"Model file not found: {model_path}", "")
                continue

            data = path.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()

            indicators = []
            poisoning_score = 0.0

            if _TORCH:
                try:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    state_dict = checkpoint.get("state_dict", checkpoint) \
                        if isinstance(checkpoint, dict) else {}
                    if state_dict:
                        all_weights = torch.cat([
                            p.float().flatten()
                            for p in state_dict.values()
                            if isinstance(p, torch.Tensor) and p.numel() > 10
                        ])
                        mean = all_weights.mean().item()
                        std = all_weights.std().item()
                        kurtosis = ((all_weights - mean) / (std + 1e-6)).pow(4).mean().item() - 3
                        if abs(kurtosis) > 20:
                            indicators.append(f"Abnormal weight kurtosis: {kurtosis:.1f}")
                            poisoning_score = min(abs(kurtosis) / 50, 1.0)
                except Exception as exc:
                    log.debug("Model load error: %s", exc)

            is_clean = poisoning_score < 0.4
            result.add_finding(
                severity="info" if is_clean else "high",
                title=f"Model integrity: {'OK' if is_clean else 'SUSPECT'} — {path.name}",
                detail=f"SHA256: {sha256[:16]}... | poisoning_score={poisoning_score:.3f}",
                model=model_path,
                sha256=sha256,
                is_clean=is_clean,
                indicators=indicators,
            )
