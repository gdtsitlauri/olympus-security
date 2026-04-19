"""Abstract base class for all OLYMPUS modules."""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from olympus.core.knowledge_base import KB
from olympus.core.logger import AUDIT, get_logger


@dataclass
class ModuleResult:
    module_id: str
    status: str                 # "success", "failure", "partial"
    findings: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error: Optional[str] = None

    def add_finding(self, severity: str, title: str, detail: str,
                    **kwargs: Any) -> None:
        self.findings.append({
            "severity": severity,
            "title": title,
            "detail": detail,
            "ts": time.time(),
            **kwargs,
        })


class BaseModule(abc.ABC):
    MODULE_ID: str = "base"
    MODULE_NAME: str = "Base Module"
    MODULE_TYPE: str = "generic"    # "offensive", "defensive", "core"

    def __init__(self) -> None:
        self.log = get_logger(self.MODULE_ID)
        self.kb = KB
        self._enabled = True

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> ModuleResult:
        """Execute the module's primary function."""

    def _start_result(self) -> tuple[ModuleResult, float]:
        result = ModuleResult(module_id=self.MODULE_ID, status="running")
        AUDIT.log(self.MODULE_ID, "start", {"module_name": self.MODULE_NAME})
        return result, time.time()

    def _finish_result(self, result: ModuleResult, t0: float) -> ModuleResult:
        result.duration_s = round(time.time() - t0, 3)
        if result.status == "running":
            result.status = "success"
        AUDIT.log(self.MODULE_ID, "finish", {
            "status": result.status,
            "findings": len(result.findings),
            "duration_s": result.duration_s,
        })
        self.kb.save()
        return result

    def health_check(self) -> dict[str, Any]:
        return {"module": self.MODULE_ID, "enabled": self._enabled, "status": "ok"}
