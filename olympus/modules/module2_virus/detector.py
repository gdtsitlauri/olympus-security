"""Module 2 — Virus Detection & Fighting (main module entry point)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.modules.module2_virus.behavioral_monitor import ProcessMonitor
from olympus.modules.module2_virus.feature_extractor import extract_features
from olympus.modules.module2_virus.ml_detector import MLDetector
from olympus.modules.module2_virus.quarantine import Quarantine


class VirusDetectionModule(BaseModule):
    MODULE_ID = "module2_virus"
    MODULE_NAME = "Virus Detection & Fighting"
    MODULE_TYPE = "defensive"

    def __init__(self) -> None:
        super().__init__()
        self._detector = MLDetector()
        self._quarantine = Quarantine()
        self._monitor = ProcessMonitor()

    def run(
        self,
        scan_path: str = ".",
        quarantine_detected: bool = False,
        behavioral: bool = False,
        extensions: list[str] | None = None,
        **kwargs: Any,
    ) -> ModuleResult:
        result, t0 = self._start_result()

        scan_root = Path(scan_path)
        if not scan_root.exists():
            result.status = "failure"
            result.error = f"Path not found: {scan_path}"
            return self._finish_result(result, t0)

        # Collect files
        if extensions is None:
            extensions = [".exe", ".dll", ".so", ".elf", ".py", ".js",
                          ".sh", ".ps1", ".bat", ".vbs", ".pdf", ".doc",
                          ".docx", ".xls", ".xlsx", ".zip", ".jar"]

        files = []
        if scan_root.is_file():
            files = [scan_root]
        else:
            for ext in extensions:
                files.extend(scan_root.rglob(f"*{ext}"))

        self.log.info("Scanning %d files in %s", len(files), scan_root)
        detected_malicious = []

        for file_path in files:
            try:
                feats = extract_features(file_path)
                detection = self._detector.detect(feats)

                finding = {
                    "file": str(file_path),
                    "sha256": feats.sha256,
                    "verdict": detection.verdict,
                    "confidence": detection.confidence,
                    "risk_score": detection.risk_score,
                    "indicators": detection.indicators,
                    "method": detection.method,
                }

                if detection.is_malicious:
                    detected_malicious.append(finding)
                    severity = "critical" if detection.risk_score > 80 else "high"
                    result.add_finding(
                        severity=severity,
                        title=f"Malware detected: {detection.verdict}",
                        detail=f"{file_path} | confidence={detection.confidence:.1%} | risk={detection.risk_score:.0f}/100",
                        file=str(file_path),
                        sha256=feats.sha256,
                        indicators=detection.indicators,
                    )
                    # Add to knowledge base
                    self.kb.add_threat(ThreatRecord(
                        id=feats.sha256[:16],
                        type="malware",
                        name=detection.verdict,
                        description="; ".join(detection.indicators[:3]),
                        severity=severity,
                        indicators=[feats.sha256, feats.md5],
                        source_module=self.MODULE_ID,
                        confidence=detection.confidence,
                    ))
                    if quarantine_detected:
                        try:
                            self._quarantine.quarantine(file_path, detection.verdict)
                            self.log.info("Quarantined: %s", file_path)
                        except Exception as exc:
                            self.log.warning("Quarantine failed for %s: %s", file_path, exc)
            except Exception as exc:
                self.log.debug("Scan error for %s: %s", file_path, exc)

        # Behavioral scan
        if behavioral:
            suspicious_procs = self._monitor.get_suspicious_processes()
            for proc in suspicious_procs:
                result.add_finding(
                    severity="high",
                    title=f"Suspicious process behavior: {proc.name}",
                    detail=f"PID {proc.pid} | events={len(proc.events)} | "
                           f"file_writes={proc.file_writes} | net_conns={proc.network_conns}",
                    pid=proc.pid,
                    process_name=proc.name,
                )

        result.metrics = {
            "files_scanned": len(files),
            "malicious_detected": len(detected_malicious),
            "detection_rate": len(detected_malicious) / max(len(files), 1),
            "quarantined": len(self._quarantine.list_quarantined()),
        }

        self.log.info("Scan complete: %d/%d malicious", len(detected_malicious), len(files))
        return self._finish_result(result, t0)

    def start_behavioral_monitor(self) -> None:
        self._monitor.start()

    def stop_behavioral_monitor(self) -> None:
        self._monitor.stop()

    def list_quarantined(self):
        return self._quarantine.list_quarantined()

    def restore_file(self, qid: str) -> bool:
        return self._quarantine.restore(qid)
