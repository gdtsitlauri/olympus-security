"""Module 10 — Forensics (timeline reconstruction, attribution, incident reporting)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.logger import AUDIT, get_logger
from olympus.modules.module4_threat_intel.threat_intel import EMBEDDED_TECHNIQUES

log = get_logger("module10")


@dataclass
class ForensicArtifact:
    artifact_id: str
    type: str                   # "file", "process", "network", "log", "registry"
    path: str
    hash_md5: str = ""
    hash_sha256: str = ""
    timestamp: float = 0.0
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chain_of_custody: list[str] = field(default_factory=list)


@dataclass
class TimelineEvent:
    ts: float
    event_type: str
    source: str
    description: str
    severity: str = "info"
    artifact_id: str = ""
    mitre_technique: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ts_str(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.ts).isoformat()


@dataclass
class AttackPath:
    steps: list[TimelineEvent]
    start_ts: float
    end_ts: float
    duration_s: float
    initial_vector: str
    techniques_used: list[str]
    estimated_actor: str = "unknown"
    confidence: float = 0.0


@dataclass
class IncidentReport:
    report_id: str
    title: str
    generated_at: float = field(default_factory=time.time)
    executive_summary: str = ""
    timeline: list[TimelineEvent] = field(default_factory=list)
    attack_path: Optional[AttackPath] = None
    artifacts: list[ForensicArtifact] = field(default_factory=list)
    iocs: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    mitre_techniques: list[str] = field(default_factory=list)
    severity: str = "unknown"

    def to_text(self) -> str:
        lines = [
            f"OLYMPUS INCIDENT REPORT — {self.report_id}",
            "=" * 60,
            f"Title: {self.title}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.generated_at))}",
            f"Severity: {self.severity.upper()}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            self.executive_summary,
            "",
            "TIMELINE OF EVENTS",
            "-" * 40,
        ]
        for ev in sorted(self.timeline, key=lambda e: e.ts):
            lines.append(f"  [{ev.ts_str}] [{ev.severity.upper():8s}] {ev.source}: {ev.description}")

        if self.attack_path:
            lines += [
                "",
                "ATTACK PATH",
                "-" * 40,
                f"Duration: {self.attack_path.duration_s:.0f}s",
                f"Initial vector: {self.attack_path.initial_vector}",
                f"Techniques: {', '.join(self.attack_path.techniques_used)}",
                f"Estimated actor: {self.attack_path.estimated_actor}",
            ]

        if self.iocs:
            lines += ["", "INDICATORS OF COMPROMISE (IOCs)", "-" * 40]
            lines.extend(f"  • {ioc}" for ioc in self.iocs)

        if self.mitre_techniques:
            lines += ["", "MITRE ATT&CK TECHNIQUES", "-" * 40]
            for tid in self.mitre_techniques:
                tech = EMBEDDED_TECHNIQUES.get(tid, {})
                lines.append(f"  • {tid}: {tech.get('name', 'Unknown')} [{tech.get('tactic', '')}]")

        if self.recommendations:
            lines += ["", "RECOMMENDATIONS", "-" * 40]
            lines.extend(f"  {i+1}. {r}" for i, r in enumerate(self.recommendations))

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "report_id": self.report_id,
            "title": self.title,
            "generated_at": self.generated_at,
            "severity": self.severity,
            "executive_summary": self.executive_summary,
            "timeline": [asdict(e) for e in self.timeline],
            "iocs": self.iocs,
            "mitre_techniques": self.mitre_techniques,
            "recommendations": self.recommendations,
            "artifact_count": len(self.artifacts),
        }, indent=2)


# ── Artifact collection ───────────────────────────────────────────────────────

def _collect_file_artifact(path: Path) -> Optional[ForensicArtifact]:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        return ForensicArtifact(
            artifact_id=hashlib.sha256(str(path).encode()).hexdigest()[:12],
            type="file",
            path=str(path),
            hash_md5=hashlib.md5(data).hexdigest(),
            hash_sha256=hashlib.sha256(data).hexdigest(),
            timestamp=path.stat().st_mtime,
            size_bytes=len(data),
            chain_of_custody=[f"Collected by OLYMPUS at {time.time():.0f}"],
        )
    except Exception:
        return None


def _read_audit_log(audit_path: Path) -> list[dict[str, Any]]:
    if not audit_path.exists():
        return []
    events = []
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _parse_syslog(syslog_path: Path) -> list[TimelineEvent]:
    events = []
    if not syslog_path.exists():
        return events
    import re
    syslog_re = re.compile(
        r'^(\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+):\s+(.+)$'
    )
    for line in syslog_path.read_text(errors="replace").splitlines():
        m = syslog_re.match(line)
        if m:
            events.append(TimelineEvent(
                ts=time.time(),   # approximate; real parser would use datetime
                event_type="syslog",
                source=m.group(3),
                description=m.group(4)[:200],
            ))
    return events[-1000:]  # limit


# ── Attribution engine ────────────────────────────────────────────────────────

def _attribute_attack(techniques: list[str]) -> tuple[str, float]:
    """Simple technique-overlap attribution to known threat groups."""
    from olympus.modules.module4_threat_intel.threat_intel import KNOWN_GROUPS
    tid_set = set(techniques)
    best_group, best_score = "Unknown", 0.0
    for group in KNOWN_GROUPS:
        overlap = len(tid_set & set(group.techniques))
        score = overlap / max(len(group.techniques), 1)
        if score > best_score:
            best_score = score
            best_group = group.name
    return best_group, round(best_score, 3)


# ── Main module ───────────────────────────────────────────────────────────────

class ForensicsModule(BaseModule):
    MODULE_ID = "module10_forensics"
    MODULE_NAME = "Digital Forensics & Incident Response"
    MODULE_TYPE = "defensive"

    def run(
        self,
        incident_path: str = ".",
        use_kb: bool = True,
        output_dir: str = "results/forensics",
        report_title: str = "OLYMPUS Incident Report",
        **kwargs: Any,
    ) -> ModuleResult:
        result, t0 = self._start_result()
        import uuid
        report_id = str(uuid.uuid4())[:8].upper()

        artifacts: list[ForensicArtifact] = []
        timeline: list[TimelineEvent] = []
        iocs: list[str] = []

        # ── collect artifacts ─────────────────────────────────────────────────
        root = Path(incident_path)
        if root.exists():
            for f in list(root.rglob("*.log"))[:50] + list(root.rglob("*.json"))[:20]:
                art = _collect_file_artifact(f)
                if art:
                    artifacts.append(art)
                    timeline.append(TimelineEvent(
                        ts=art.timestamp,
                        event_type="file",
                        source="filesystem",
                        description=f"File: {f.name} ({art.size_bytes} bytes)",
                        artifact_id=art.artifact_id,
                    ))

        # ── parse OLYMPUS audit log ───────────────────────────────────────────
        from olympus.core.config import CONFIG
        audit_events = _read_audit_log(CONFIG.audit_log_path)
        for ev in audit_events:
            severity = ev.get("severity", "INFO").lower()
            timeline.append(TimelineEvent(
                ts=time.mktime(time.strptime(ev["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
                   if "ts" in ev else time.time(),
                event_type="olympus_audit",
                source=ev.get("module", "unknown"),
                description=f"{ev.get('action', '')} — {json.dumps({k: v for k, v in ev.items() if k not in ('ts', 'module', 'action')})}",
                severity="high" if severity in ("high", "critical") else "info",
            ))

        # ── pull from knowledge base ──────────────────────────────────────────
        techniques_seen: list[str] = []
        if use_kb:
            threats = self.kb.query_threats(limit=200)
            for threat in threats:
                iocs.extend(threat.indicators)
                techniques_seen.extend(threat.mitre_techniques)
                timeline.append(TimelineEvent(
                    ts=threat.first_seen,
                    event_type="threat",
                    source=threat.source_module,
                    description=f"{threat.type}: {threat.name} (severity={threat.severity})",
                    severity=threat.severity,
                    mitre_technique=threat.mitre_techniques[0] if threat.mitre_techniques else "",
                ))

        techniques_seen = list(dict.fromkeys(techniques_seen))  # dedup preserve order
        iocs = list(set(iocs))[:100]

        # ── reconstruct attack path ───────────────────────────────────────────
        sorted_timeline = sorted(timeline, key=lambda e: e.ts)
        attack_events = [e for e in sorted_timeline if e.severity in ("high", "critical")]

        attack_path = None
        if attack_events:
            actor, confidence = _attribute_attack(techniques_seen)
            attack_path = AttackPath(
                steps=attack_events,
                start_ts=attack_events[0].ts if attack_events else time.time(),
                end_ts=attack_events[-1].ts if attack_events else time.time(),
                duration_s=(attack_events[-1].ts - attack_events[0].ts) if len(attack_events) > 1 else 0,
                initial_vector=attack_events[0].description[:100] if attack_events else "unknown",
                techniques_used=techniques_seen[:10],
                estimated_actor=actor,
                confidence=confidence,
            )

        # ── determine overall severity ────────────────────────────────────────
        sev_counts = defaultdict(int)
        for e in timeline:
            sev_counts[e.severity] += 1
        overall_severity = (
            "critical" if sev_counts.get("critical", 0) > 0 else
            "high" if sev_counts.get("high", 0) > 0 else
            "medium" if sev_counts.get("medium", 0) > 0 else
            "low"
        )

        # ── recommendations ───────────────────────────────────────────────────
        recommendations = self._generate_recommendations(techniques_seen, overall_severity)

        # ── executive summary ─────────────────────────────────────────────────
        exec_summary = (
            f"OLYMPUS detected {len(timeline)} events across {len(artifacts)} artifacts. "
            f"Overall severity: {overall_severity.upper()}. "
            f"{len(techniques_seen)} MITRE ATT&CK techniques observed. "
        )
        if attack_path:
            exec_summary += (
                f"Attack duration: {attack_path.duration_s:.0f}s. "
                f"Estimated actor: {attack_path.estimated_actor} (confidence={attack_path.confidence:.0%}). "
            )
        exec_summary += f"{len(iocs)} indicators of compromise identified."

        # ── build report ──────────────────────────────────────────────────────
        report = IncidentReport(
            report_id=report_id,
            title=report_title,
            executive_summary=exec_summary,
            timeline=sorted_timeline[:200],
            attack_path=attack_path,
            artifacts=artifacts,
            iocs=iocs,
            recommendations=recommendations,
            mitre_techniques=techniques_seen[:20],
            severity=overall_severity,
        )

        # ── save reports ──────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{report_id}.txt").write_text(report.to_text())
        (out / f"report_{report_id}.json").write_text(report.to_json())

        # ── findings ──────────────────────────────────────────────────────────
        result.add_finding(
            severity=overall_severity,
            title=f"Incident report generated: {report_id}",
            detail=exec_summary,
            report_id=report_id,
            output_files=[str(out / f"report_{report_id}.txt"),
                          str(out / f"report_{report_id}.json")],
        )

        if attack_path and attack_path.estimated_actor != "Unknown":
            result.add_finding(
                severity="high",
                title=f"Threat actor attribution: {attack_path.estimated_actor}",
                detail=f"Confidence: {attack_path.confidence:.0%} based on technique overlap",
                actor=attack_path.estimated_actor,
                techniques=techniques_seen,
            )

        result.metrics = {
            "timeline_events": len(timeline),
            "artifacts_collected": len(artifacts),
            "iocs_found": len(iocs),
            "techniques_identified": len(techniques_seen),
            "overall_severity": overall_severity,
            "attack_duration_s": attack_path.duration_s if attack_path else 0,
        }

        self.log.info("Forensics complete: %d events, severity=%s, actor=%s",
                      len(timeline), overall_severity,
                      attack_path.estimated_actor if attack_path else "unknown")
        return self._finish_result(result, t0)

    def _generate_recommendations(self, techniques: list[str], severity: str) -> list[str]:
        recs = [
            "Preserve all logs and artifacts for chain of custody",
            "Reset all potentially compromised credentials immediately",
            "Isolate affected systems from network",
            "Notify relevant stakeholders per incident response plan",
        ]
        tactic_recs = {
            "T1059": "Restrict PowerShell and script interpreter access; enable script block logging",
            "T1078": "Audit all accounts; enforce MFA; review privileged access",
            "T1486": "Restore from offline backups; do not pay ransom",
            "T1566": "Block sender domains; enhance email filtering; user awareness training",
            "T1055": "Deploy endpoint detection with process injection detection",
            "T1003": "Enable LSASS protection; audit credential access events",
        }
        for tid in techniques[:5]:
            if tid in tactic_recs:
                recs.append(tactic_recs[tid])
        if severity == "critical":
            recs.insert(0, "CRITICAL: Activate full incident response team immediately")
            recs.insert(1, "Consider engaging external incident response firm")
        return recs
