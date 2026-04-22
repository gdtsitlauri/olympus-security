"""OLYMPUS-SECURITY package bootstrap."""

from __future__ import annotations

from importlib import import_module

from olympus.core.knowledge_base import KB
from olympus.core.logger import get_logger
from olympus.core.orchestrator import ORCHESTRATOR

__version__ = "1.0.0"
__author__ = "George David Tsitlauri"
__license__ = "MIT"

log = get_logger("olympus.bootstrap")

_MODULE_REGISTRY = [
    ("olympus.modules.module1_pentest", "PentestModule"),
    ("olympus.modules.module2_virus", "VirusDetectionModule"),
    ("olympus.modules.module3_zeroday", "ZeroDayModule"),
    ("olympus.modules.module4_threat_intel", "ThreatIntelligenceModule"),
    ("olympus.modules.module5_deception", "DeceptionModule"),
    ("olympus.modules.module6_evolution", "SelfEvolutionModule"),
    ("olympus.modules.module7_social_eng", "SocialEngDetectionModule"),
    ("olympus.modules.module8_ai_integrity", "AIIntegrityModule"),
    ("olympus.modules.module9_llm_defense", "LLMDefenseModule"),
    ("olympus.modules.module10_forensics", "ForensicsModule"),
]


def _register_all() -> None:
    """Best-effort auto-registration for available modules.

    OLYMPUS contains optional or heavyweight module dependencies. Importing the
    package should not fail just because one unrelated module cannot initialize in
    the current environment.
    """

    for module_path, class_name in _MODULE_REGISTRY:
        try:
            module = import_module(module_path)
            module_class = getattr(module, class_name)
            ORCHESTRATOR.register(module_class())
        except Exception as exc:  # pragma: no cover - defensive bootstrap path
            log.warning("Skipping auto-registration for %s.%s: %s", module_path, class_name, exc)


_register_all()

__all__ = ["ORCHESTRATOR", "KB", "__version__"]
