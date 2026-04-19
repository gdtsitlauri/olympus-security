"""Structured audit + runtime logging for OLYMPUS."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olympus.core.config import CONFIG


def _make_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(name: str) -> logging.Logger:
    return _make_logger(f"olympus.{name}")


class AuditLogger:
    """Append-only JSONL audit trail of all OLYMPUS actions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CONFIG.audit_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, module: str, action: str, details: dict[str, Any], severity: str = "INFO") -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "action": action,
            "severity": severity,
            **details,
        }
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries


AUDIT = AuditLogger()
