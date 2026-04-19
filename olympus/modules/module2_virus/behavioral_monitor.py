"""Behavioral monitoring — system call tracing, process anomaly detection."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from olympus.core.logger import get_logger

log = get_logger("module2.behavioral")


@dataclass
class ProcessEvent:
    pid: int
    name: str
    event_type: str         # "create", "file_write", "network", "registry", "terminate"
    detail: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "info"  # "info", "suspicious", "malicious"


@dataclass
class BehaviorProfile:
    pid: int
    name: str
    events: list[ProcessEvent] = field(default_factory=list)
    file_writes: int = 0
    network_conns: int = 0
    child_processes: int = 0
    registry_writes: int = 0
    anomaly_score: float = 0.0


# Suspicious behavioral patterns
_SUSPICIOUS_PATTERNS = [
    # Ransomware: mass file modification
    ("mass_file_modify", lambda p: p.file_writes > 100, "high",
     "Process modified >100 files — possible ransomware"),
    # Crypto-miner: sustained high CPU
    ("high_cpu_sustained", lambda p: p.anomaly_score > 0.7, "medium",
     "Sustained anomalous resource usage"),
    # C2: frequent network connections
    ("c2_beacon", lambda p: p.network_conns > 50, "high",
     "Excessive outbound connections — possible C2 beaconing"),
    # Dropper: spawned many processes
    ("process_spray", lambda p: p.child_processes > 10, "medium",
     "Process spawned many children — possible dropper"),
]

_SENSITIVE_PATHS_REGEX = re.compile(
    r"(passwd|shadow|\.ssh|id_rsa|\.aws|credentials|wallet|\.env|"
    r"chrome.*(login|cookies)|firefox.*key\d\.db)",
    re.I,
)

_C2_PORTS = {4444, 1337, 8080, 8443, 443, 6667, 6668}   # common C2 / IRC ports


class ProcessMonitor:
    """Monitor running processes for malicious behavior (Linux/Windows)."""

    def __init__(self) -> None:
        self._profiles: dict[int, BehaviorProfile] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[ProcessEvent], None]] = []

    def start(self, interval: float = 2.0) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._thread.start()
        log.info("Process monitor started (interval=%.1fs)", interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Process monitor stopped")

    def on_event(self, callback: Callable[[ProcessEvent], None]) -> None:
        self._callbacks.append(callback)

    def _emit(self, event: ProcessEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.warning("Event callback error: %s", exc)

    def _monitor_loop(self, interval: float) -> None:
        while self._running:
            self._collect_snapshot()
            time.sleep(interval)

    def _collect_snapshot(self) -> None:
        try:
            if os.name == "nt":
                self._collect_windows()
            else:
                self._collect_linux()
        except Exception as exc:
            log.debug("Snapshot error: %s", exc)

    def _collect_linux(self) -> None:
        proc_dir = Path("/proc")
        for pid_dir in proc_dir.iterdir():
            if not pid_dir.name.isdigit():
                continue
            pid = int(pid_dir.name)
            try:
                comm = (pid_dir / "comm").read_text().strip()
                with self._lock:
                    if pid not in self._profiles:
                        self._profiles[pid] = BehaviorProfile(pid=pid, name=comm)
                        self._emit(ProcessEvent(pid=pid, name=comm,
                                                event_type="create", detail="Process started"))

                # Check open file descriptors for sensitive paths
                fd_dir = pid_dir / "fd"
                if fd_dir.exists():
                    for fd in fd_dir.iterdir():
                        try:
                            target = str(fd.resolve())
                            if _SENSITIVE_PATHS_REGEX.search(target):
                                self._emit(ProcessEvent(
                                    pid=pid, name=comm,
                                    event_type="file_write",
                                    detail=f"Access to sensitive path: {target}",
                                    severity="suspicious",
                                ))
                        except Exception:
                            pass

                # Check network connections
                net = pid_dir / "net" / "tcp"
                if net.exists():
                    lines = net.read_text().splitlines()[1:]
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4 and parts[3] == "01":  # ESTABLISHED
                            rem = parts[2]
                            port = int(rem.split(":")[1], 16)
                            if port in _C2_PORTS:
                                self._emit(ProcessEvent(
                                    pid=pid, name=comm,
                                    event_type="network",
                                    detail=f"Connection on suspicious port {port}",
                                    severity="suspicious",
                                ))

            except (PermissionError, FileNotFoundError, ProcessLookupError):
                pass

    def _collect_windows(self) -> None:
        try:
            out = subprocess.check_output(
                ["tasklist", "/fo", "csv", "/nh"],
                capture_output=False, text=True, timeout=5
            )
            for line in out.splitlines():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        pid = int(parts[1])
                        with self._lock:
                            if pid not in self._profiles:
                                self._profiles[pid] = BehaviorProfile(pid=pid, name=name)
                    except ValueError:
                        pass
        except Exception:
            pass

    def get_suspicious_processes(self) -> list[BehaviorProfile]:
        with self._lock:
            suspicious = []
            for profile in self._profiles.values():
                for _, check_fn, _, _ in _SUSPICIOUS_PATTERNS:
                    if check_fn(profile):
                        suspicious.append(profile)
                        break
            return suspicious

    def get_all_profiles(self) -> list[BehaviorProfile]:
        with self._lock:
            return list(self._profiles.values())


class NetworkAnomalyDetector:
    """Detect anomalous network traffic patterns using statistical baseline."""

    def __init__(self, window: int = 60) -> None:
        self._window = window
        self._baseline: dict[str, float] = {}
        self._history: dict[str, list[float]] = defaultdict(list)

    def update(self, metric: str, value: float) -> Optional[float]:
        """Update metric; return anomaly z-score if anomalous, else None."""
        history = self._history[metric]
        history.append(value)
        if len(history) > self._window:
            history.pop(0)
        if len(history) < 10:
            return None

        import statistics
        mean = statistics.mean(history)
        stdev = statistics.stdev(history) or 1e-6
        z = (value - mean) / stdev
        return z if abs(z) > 3.0 else None

    def detect_c2_pattern(self, intervals: list[float]) -> float:
        """Score regularity of beacon intervals (C2 typically regular)."""
        if len(intervals) < 5:
            return 0.0
        import statistics
        cv = statistics.stdev(intervals) / (statistics.mean(intervals) or 1e-6)
        # Low CV = very regular = suspicious C2 pattern
        regularity_score = max(0.0, 1.0 - cv)
        return round(regularity_score, 4)
