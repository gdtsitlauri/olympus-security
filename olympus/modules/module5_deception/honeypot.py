"""Module 5 — Deception (honeypots, fake services, attacker tracking)."""

from __future__ import annotations

import json
import random
import socket
import string
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import ThreatRecord
from olympus.core.logger import AUDIT, get_logger

log = get_logger("module5")


# ── Attacker event ─────────────────────────────────────────────────────────────

@dataclass
class AttackerEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    honeypot_id: str = ""
    attacker_ip: str = ""
    attacker_port: int = 0
    service: str = ""
    timestamp: float = field(default_factory=time.time)
    data: bytes = b""
    data_decoded: str = ""
    event_type: str = "connection"  # connection, auth_attempt, command, scan
    credentials_attempted: Optional[tuple[str, str]] = None


@dataclass
class AttackerProfile:
    ip: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    events: list[AttackerEvent] = field(default_factory=list)
    total_connections: int = 0
    services_probed: list[str] = field(default_factory=list)
    credentials_tried: list[tuple[str, str]] = field(default_factory=list)
    ttps: list[str] = field(default_factory=list)
    threat_score: float = 0.0


# ── Fake credential/data generators ──────────────────────────────────────────

def _fake_ssh_banner() -> str:
    versions = ["OpenSSH_8.2p1 Ubuntu-4ubuntu0.5", "OpenSSH_7.4p1", "OpenSSH_9.0"]
    return f"SSH-2.0-{random.choice(versions)}\r\n"


def _fake_http_response(path: str = "/") -> bytes:
    body = f"""<!DOCTYPE html><html><head><title>Admin Panel</title></head>
<body><h1>System Administration</h1>
<form method="POST" action="/login">
  <input name="username" placeholder="Username"><br>
  <input name="password" type="password" placeholder="Password"><br>
  <input type="submit" value="Login">
</form></body></html>"""
    return (
        f"HTTP/1.1 200 OK\r\n"
        f"Server: Apache/2.4.51\r\n"
        f"Content-Type: text/html\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"X-Powered-By: PHP/7.4.33\r\n"
        f"\r\n{body}"
    ).encode()


def _fake_ftp_banner() -> str:
    return "220 Microsoft FTP Service\r\n"


def _fake_mysql_error() -> bytes:
    # MySQL initial handshake packet simulation
    return b"\x4a\x00\x00\x00\x0a" + b"8.0.27\x00" + b"\x01\x00\x00\x00" + b"\x41" * 8 + b"\x00"


def _fake_credentials() -> dict[str, Any]:
    """Generate fake but plausible-looking credentials for honeypot lures."""
    return {
        "admin_user": "admin",
        "admin_pass": "Admin@" + "".join(random.choices(string.digits, k=4)),
        "db_user": "dbuser",
        "db_pass": "DbPass" + "".join(random.choices(string.digits, k=4)),
        "api_key": "sk-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=32)),
    }


# ── Honeypot service base ─────────────────────────────────────────────────────

class HoneypotService:
    def __init__(self, port: int, service: str,
                 on_event: Callable[[AttackerEvent], None]) -> None:
        self.honeypot_id = str(uuid.uuid4())[:8]
        self.port = port
        self.service = service
        self._on_event = on_event
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind(("0.0.0.0", self.port))
            self._server.listen(10)
            self._server.settimeout(1.0)
            self._running = True
            self._thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._thread.start()
            log.info("Honeypot %s started on port %d (%s)", self.honeypot_id, self.port, self.service)
        except OSError as exc:
            log.warning("Could not bind port %d: %s", self.port, exc)

    def stop(self) -> None:
        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server.accept()
                t = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    log.debug("Accept error")

    def _handle(self, conn: socket.socket, addr: tuple) -> None:
        ip, port = addr
        event = AttackerEvent(
            honeypot_id=self.honeypot_id,
            attacker_ip=ip,
            attacker_port=port,
            service=self.service,
        )
        try:
            banner = self._get_banner()
            if banner:
                conn.sendall(banner if isinstance(banner, bytes) else banner.encode())

            conn.settimeout(10.0)
            try:
                data = conn.recv(4096)
                event.data = data
                event.data_decoded = data.decode("utf-8", errors="replace")[:500]
                event.event_type = self._classify_data(data)
                event.credentials_attempted = self._extract_credentials(data)
            except socket.timeout:
                pass

            resp = self._fake_response(data if "data" in dir() else b"")
            if resp:
                try:
                    conn.sendall(resp)
                except Exception:
                    pass
        except Exception as exc:
            log.debug("Honeypot handler error: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self._on_event(event)

    def _get_banner(self):
        banners = {
            "ssh": _fake_ssh_banner(),
            "http": None,
            "ftp": _fake_ftp_banner(),
            "mysql": _fake_mysql_error(),
        }
        return banners.get(self.service)

    def _fake_response(self, data: bytes) -> Optional[bytes]:
        decoded = data.decode("utf-8", errors="replace").lower()
        if self.service == "http":
            return _fake_http_response()
        if self.service == "ftp" and "user" in decoded:
            return b"331 Password required\r\n"
        if self.service == "ssh":
            return None  # SSH handled by banner
        return b"500 Internal Server Error\r\n"

    def _classify_data(self, data: bytes) -> str:
        decoded = data.decode("utf-8", errors="replace").lower()
        if any(w in decoded for w in ["user", "username", "login", "password", "pass"]):
            return "auth_attempt"
        if any(w in decoded for w in ["get ", "post ", "head "]):
            return "http_request"
        if len(data) < 5:
            return "scan"
        return "data"

    def _extract_credentials(self, data: bytes) -> Optional[tuple[str, str]]:
        decoded = data.decode("utf-8", errors="replace")
        import re
        # HTTP basic auth
        basic = re.search(r"Authorization: Basic ([A-Za-z0-9+/=]+)", decoded, re.I)
        if basic:
            import base64
            try:
                creds = base64.b64decode(basic.group(1)).decode()
                user, _, pwd = creds.partition(":")
                return (user, pwd)
            except Exception:
                pass
        # form data
        form = re.search(r"username=([^&]+)&password=([^&\s]+)", decoded, re.I)
        if form:
            return (form.group(1), form.group(2))
        return None


# ── Main module ───────────────────────────────────────────────────────────────

class DeceptionModule(BaseModule):
    MODULE_ID = "module5_deception"
    MODULE_NAME = "Deception & Honeypots"
    MODULE_TYPE = "defensive"

    def __init__(self) -> None:
        super().__init__()
        self._services: list[HoneypotService] = []
        self._attacker_profiles: dict[str, AttackerProfile] = {}
        self._events: list[AttackerEvent] = []
        self._events_lock = threading.Lock()

    def run(
        self,
        services: dict[str, int] | None = None,
        duration: float = 60.0,
        analyze_existing: bool = False,
        **kwargs: Any,
    ) -> ModuleResult:
        """
        Args:
            services: {service_type: port} e.g. {"ssh": 2222, "http": 8080}
            duration: How many seconds to run honeypots (0 = persistent background)
            analyze_existing: Just analyze captured events without starting new services
        """
        result, t0 = self._start_result()

        if analyze_existing:
            self._compile_findings(result)
            return self._finish_result(result, t0)

        if services is None:
            services = {"ssh": 2222, "http": 8888, "ftp": 2121}

        # Start honeypot services
        for service_type, port in services.items():
            hp = HoneypotService(port, service_type, self._record_event)
            hp.start()
            self._services.append(hp)
            result.add_finding(
                severity="info",
                title=f"Honeypot started: {service_type} on port {port}",
                detail=f"Honeypot ID: {hp.honeypot_id}",
                honeypot_id=hp.honeypot_id,
                port=port,
            )

        # Generate lure files/credentials
        lures = _fake_credentials()
        result.add_finding(
            severity="info",
            title="Deception lures deployed",
            detail=f"Fake credentials and data generated: {list(lures.keys())}",
            lures=lures,
        )

        if duration > 0:
            self.log.info("Honeypots active for %.0fs...", duration)
            time.sleep(duration)
            self._stop_all()
            self._compile_findings(result)

        result.metrics = {
            "honeypots_deployed": len(self._services),
            "attackers_detected": len(self._attacker_profiles),
            "total_events": len(self._events),
            "credentials_captured": sum(
                1 for e in self._events if e.credentials_attempted
            ),
        }

        return self._finish_result(result, t0)

    def _record_event(self, event: AttackerEvent) -> None:
        with self._events_lock:
            self._events.append(event)

        ip = event.attacker_ip
        if ip not in self._attacker_profiles:
            self._attacker_profiles[ip] = AttackerProfile(ip=ip)
        profile = self._attacker_profiles[ip]
        profile.last_seen = time.time()
        profile.total_connections += 1
        profile.events.append(event)
        if event.service not in profile.services_probed:
            profile.services_probed.append(event.service)
        if event.credentials_attempted:
            profile.credentials_tried.append(event.credentials_attempted)

        # Update threat score
        profile.threat_score = min(
            profile.total_connections * 0.1 +
            len(profile.services_probed) * 0.2 +
            len(profile.credentials_tried) * 0.3,
            1.0,
        )

        AUDIT.log("module5_deception", "attacker_event", {
            "ip": ip,
            "service": event.service,
            "type": event.event_type,
            "threat_score": profile.threat_score,
        }, severity="HIGH")

        # Feed to KB
        self.kb.add_threat(ThreatRecord(
            id=f"attacker-{ip.replace('.', '_')}",
            type="ioc",
            name=f"Attacker IP: {ip}",
            description=f"Probed services: {profile.services_probed}",
            severity="high" if profile.threat_score > 0.5 else "medium",
            indicators=[ip],
            source_module=self.MODULE_ID,
            confidence=profile.threat_score,
        ))

    def _compile_findings(self, result: ModuleResult) -> None:
        for ip, profile in self._attacker_profiles.items():
            severity = "critical" if profile.threat_score > 0.8 else \
                       "high" if profile.threat_score > 0.5 else "medium"
            result.add_finding(
                severity=severity,
                title=f"Attacker profiled: {ip}",
                detail=f"Connections: {profile.total_connections} | "
                       f"Services: {profile.services_probed} | "
                       f"Creds tried: {len(profile.credentials_tried)} | "
                       f"Threat score: {profile.threat_score:.2f}",
                attacker_ip=ip,
                threat_score=profile.threat_score,
                credentials_tried=[list(c) for c in profile.credentials_tried],
            )

    def _stop_all(self) -> None:
        for svc in self._services:
            svc.stop()
        self._services.clear()

    def get_attacker_profiles(self) -> list[AttackerProfile]:
        return list(self._attacker_profiles.values())
