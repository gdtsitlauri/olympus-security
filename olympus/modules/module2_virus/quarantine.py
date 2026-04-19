"""Quarantine engine — isolate, hash-verify, and restore infected files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from olympus.core.config import CONFIG
from olympus.core.logger import AUDIT, get_logger

log = get_logger("module2.quarantine")

_QUARANTINE_DIR = CONFIG.project_root / "data" / "quarantine"
_MANIFEST = _QUARANTINE_DIR / "manifest.json"


@dataclass
class QuarantineRecord:
    qid: str
    original_path: str
    quarantine_path: str
    sha256: str
    verdict: str
    timestamp: float = field(default_factory=time.time)
    restored: bool = False
    restored_at: Optional[float] = None


class Quarantine:
    def __init__(self) -> None:
        _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, QuarantineRecord] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if _MANIFEST.exists():
            try:
                data = json.loads(_MANIFEST.read_text())
                for r in data:
                    rec = QuarantineRecord(**r)
                    self._records[rec.qid] = rec
            except Exception as exc:
                log.warning("Manifest load error: %s", exc)

    def _save_manifest(self) -> None:
        data = [asdict(r) for r in self._records.values()]
        _MANIFEST.write_text(json.dumps(data, indent=2))

    def quarantine(self, path: str | Path, verdict: str) -> QuarantineRecord:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot quarantine: {path}")

        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        qid = sha256[:16]
        q_path = _QUARANTINE_DIR / f"{qid}.quar"

        # XOR-obfuscate quarantined file (prevent accidental execution)
        obfuscated = bytes(b ^ 0xAA for b in data)
        q_path.write_bytes(obfuscated)

        # Remove original
        path.unlink()

        record = QuarantineRecord(
            qid=qid,
            original_path=str(path),
            quarantine_path=str(q_path),
            sha256=sha256,
            verdict=verdict,
        )
        self._records[qid] = record
        self._save_manifest()

        AUDIT.log("module2_quarantine", "quarantine", {
            "qid": qid, "path": str(path), "verdict": verdict,
        }, severity="HIGH")
        log.info("Quarantined %s → %s (verdict: %s)", path, q_path, verdict)
        return record

    def restore(self, qid: str, target_path: Optional[str] = None) -> bool:
        record = self._records.get(qid)
        if not record:
            log.error("QID not found: %s", qid)
            return False

        q_path = Path(record.quarantine_path)
        if not q_path.exists():
            log.error("Quarantine file missing: %s", q_path)
            return False

        obfuscated = q_path.read_bytes()
        original_data = bytes(b ^ 0xAA for b in obfuscated)

        # Verify integrity
        if hashlib.sha256(original_data).hexdigest() != record.sha256:
            log.error("Integrity check FAILED for %s", qid)
            return False

        dest = Path(target_path or record.original_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(original_data)
        record.restored = True
        record.restored_at = time.time()
        self._save_manifest()

        AUDIT.log("module2_quarantine", "restore", {"qid": qid, "dest": str(dest)})
        log.info("Restored %s → %s", qid, dest)
        return True

    def list_quarantined(self) -> list[QuarantineRecord]:
        return [r for r in self._records.values() if not r.restored]

    def delete_permanently(self, qid: str) -> bool:
        record = self._records.get(qid)
        if not record:
            return False
        q_path = Path(record.quarantine_path)
        if q_path.exists():
            q_path.unlink()
        del self._records[qid]
        self._save_manifest()
        AUDIT.log("module2_quarantine", "delete", {"qid": qid}, severity="HIGH")
        return True
