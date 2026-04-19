"""OLYMPUS global configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent


@dataclass
class GPUConfig:
    enabled: bool = True
    device: str = "cuda"
    vram_budget_gb: float = 3.5          # GTX 1650 safe budget
    cuda_version: str = "12.x"
    fallback_cpu: bool = True


@dataclass
class SandboxConfig:
    enabled: bool = True
    docker_image: str = "olympus-sandbox:latest"
    network_isolated: bool = True
    timeout_seconds: int = 300
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 2048


@dataclass
class KnowledgeBaseConfig:
    path: Path = ROOT / "data" / "knowledge_base.json"
    mitre_attack_path: Path = ROOT / "data" / "mitre_attack.json"
    max_entries: int = 100_000


@dataclass
class ModuleConfig:
    enabled: bool = True
    log_level: str = "INFO"
    audit_log: bool = True


@dataclass
class OlympusConfig:
    project_root: Path = ROOT
    gpu: GPUConfig = field(default_factory=GPUConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)

    # Module toggles
    modules: dict[str, ModuleConfig] = field(default_factory=lambda: {
        f"module{i}": ModuleConfig() for i in range(1, 11)
    })

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Dashboard
    dashboard_port: int = 8501

    # Audit log
    audit_log_path: Path = ROOT / "data" / "audit.jsonl"

    # Experiment settings
    random_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    confidence_level: float = 0.95

    @classmethod
    def from_env(cls) -> "OlympusConfig":
        cfg = cls()
        cfg.gpu.enabled = os.getenv("OLYMPUS_GPU", "1") == "1"
        cfg.gpu.device = os.getenv("OLYMPUS_DEVICE", "cuda")
        cfg.api_port = int(os.getenv("OLYMPUS_API_PORT", "8000"))
        cfg.dashboard_port = int(os.getenv("OLYMPUS_DASHBOARD_PORT", "8501"))
        return cfg


CONFIG = OlympusConfig.from_env()
