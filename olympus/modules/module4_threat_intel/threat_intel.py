"""Module 4 — Threat Intelligence (TTP learning, MITRE ATT&CK, prediction)."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.knowledge_base import AttackPattern, ThreatRecord, KB
from olympus.core.logger import get_logger

log = get_logger("module4")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ── MITRE ATT&CK data ─────────────────────────────────────────────────────────

MITRE_TACTICS = [
    "reconnaissance", "resource-development", "initial-access",
    "execution", "persistence", "privilege-escalation",
    "defense-evasion", "credential-access", "discovery",
    "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact",
]

# Embedded subset of MITRE ATT&CK (T1059, T1078, etc.) for offline use
EMBEDDED_TECHNIQUES: dict[str, dict[str, str]] = {
    "T1059": {"name": "Command and Scripting Interpreter", "tactic": "execution"},
    "T1059.001": {"name": "PowerShell", "tactic": "execution"},
    "T1059.003": {"name": "Windows Command Shell", "tactic": "execution"},
    "T1078": {"name": "Valid Accounts", "tactic": "initial-access"},
    "T1190": {"name": "Exploit Public-Facing Application", "tactic": "initial-access"},
    "T1566": {"name": "Phishing", "tactic": "initial-access"},
    "T1566.001": {"name": "Spearphishing Attachment", "tactic": "initial-access"},
    "T1053": {"name": "Scheduled Task/Job", "tactic": "persistence"},
    "T1547": {"name": "Boot or Logon Autostart Execution", "tactic": "persistence"},
    "T1055": {"name": "Process Injection", "tactic": "privilege-escalation"},
    "T1068": {"name": "Exploitation for Privilege Escalation", "tactic": "privilege-escalation"},
    "T1562": {"name": "Impair Defenses", "tactic": "defense-evasion"},
    "T1027": {"name": "Obfuscated Files or Information", "tactic": "defense-evasion"},
    "T1003": {"name": "OS Credential Dumping", "tactic": "credential-access"},
    "T1082": {"name": "System Information Discovery", "tactic": "discovery"},
    "T1046": {"name": "Network Service Discovery", "tactic": "discovery"},
    "T1021": {"name": "Remote Services", "tactic": "lateral-movement"},
    "T1041": {"name": "Exfiltration Over C2 Channel", "tactic": "exfiltration"},
    "T1486": {"name": "Data Encrypted for Impact", "tactic": "impact"},
    "T1498": {"name": "Network Denial of Service", "tactic": "impact"},
    "T1071": {"name": "Application Layer Protocol (C2)", "tactic": "command-and-control"},
    "T1095": {"name": "Non-Application Layer Protocol (C2)", "tactic": "command-and-control"},
    "T1057": {"name": "Process Discovery", "tactic": "discovery"},
    "T1005": {"name": "Data from Local System", "tactic": "collection"},
    "T1560": {"name": "Archive Collected Data", "tactic": "collection"},
}


# ── TTP sequence predictor ────────────────────────────────────────────────────

class TTPSequencePredictor(nn.Module):  # type: ignore[misc]
    """LSTM that predicts the next likely ATT&CK technique given an observed sequence."""

    def __init__(self, vocab_size: int = len(EMBEDDED_TECHNIQUES) + 2,
                 embed_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])   # predict next from last hidden state


@dataclass
class TTPPrediction:
    predicted_techniques: list[str]
    probabilities: dict[str, float]
    next_likely_tactic: str
    confidence: float
    kill_chain_position: int


@dataclass
class ThreatGroup:
    name: str
    aliases: list[str]
    techniques: list[str]
    sectors_targeted: list[str]
    motivation: str = "unknown"
    sophistication: str = "medium"


# Known threat groups (simplified subset)
KNOWN_GROUPS: list[ThreatGroup] = [
    ThreatGroup("APT28", ["Fancy Bear", "Sofacy"], ["T1566", "T1059", "T1027"], ["government", "military"]),
    ThreatGroup("APT29", ["Cozy Bear"], ["T1566", "T1078", "T1003"], ["government", "healthcare"]),
    ThreatGroup("Lazarus", ["Hidden Cobra"], ["T1486", "T1059", "T1041"], ["financial", "crypto"]),
    ThreatGroup("REvil", ["Sodinokibi"], ["T1486", "T1078", "T1021"], ["enterprise", "government"]),
    ThreatGroup("FIN7", [], ["T1566", "T1059", "T1003"], ["retail", "restaurant", "financial"]),
]


class ThreatIntelligenceModule(BaseModule):
    MODULE_ID = "module4_threat_intel"
    MODULE_NAME = "Threat Intelligence"
    MODULE_TYPE = "defensive"

    def __init__(self) -> None:
        super().__init__()
        self._technique_index = {t: i + 1 for i, t in enumerate(EMBEDDED_TECHNIQUES)}
        self._predictor: Optional[TTPSequencePredictor] = None
        self._device = None

        if _TORCH:
            from olympus.core.device import get_device
            self._device = get_device()
            self._predictor = TTPSequencePredictor(
                vocab_size=len(EMBEDDED_TECHNIQUES) + 2
            ).to(self._device)
            self._predictor.eval()
            log.info("TTP sequence predictor on %s", self._device)

    def run(
        self,
        observed_techniques: Optional[list[str]] = None,
        analyze_kb: bool = True,
        predict_next: bool = True,
        **kwargs: Any,
    ) -> ModuleResult:
        result, t0 = self._start_result()

        techniques = list(observed_techniques or [])

        # ── enrich from knowledge base ────────────────────────────────────────
        if analyze_kb:
            kb_patterns = self.kb.get_attack_patterns()
            kb_techniques = [p.technique_id for p in kb_patterns if p.technique_id]
            techniques = list(set(techniques + kb_techniques))
            self.log.info("Loaded %d techniques from KB", len(kb_techniques))

        if not techniques:
            techniques = ["T1566", "T1059", "T1027"]  # default example chain
            self.log.info("Using default example technique chain")

        # ── TTP analysis ──────────────────────────────────────────────────────
        tactic_counts = Counter()
        for tid in techniques:
            tech = EMBEDDED_TECHNIQUES.get(tid)
            if tech:
                tactic_counts[tech["tactic"]] += 1

        kill_chain_stage = self._estimate_kill_chain_stage(tactic_counts)

        # ── threat group attribution ──────────────────────────────────────────
        attributed_groups = self._attribute_to_groups(techniques)

        for group in attributed_groups:
            result.add_finding(
                severity="high",
                title=f"Threat group attribution: {group.name}",
                detail=f"Matched {len(set(techniques) & set(group.techniques))} techniques. "
                       f"Motivation: {group.motivation}. Sectors: {', '.join(group.sectors_targeted)}",
                group=group.name,
                aliases=group.aliases,
                matched_techniques=list(set(techniques) & set(group.techniques)),
            )

        # ── prediction ────────────────────────────────────────────────────────
        prediction = None
        if predict_next and techniques:
            prediction = self._predict_next(techniques)
            result.add_finding(
                severity="medium",
                title=f"Predicted next attack vector: {prediction.next_likely_tactic}",
                detail=f"Top techniques: {', '.join(prediction.predicted_techniques[:3])}. "
                       f"Confidence: {prediction.confidence:.1%}",
                predicted_techniques=prediction.predicted_techniques[:5],
                kill_chain_position=prediction.kill_chain_position,
            )

        # ── auto-generate signatures ──────────────────────────────────────────
        signatures = self._generate_signatures(techniques)
        for sig in signatures:
            result.add_finding(
                severity="info",
                title=f"Auto-generated detection signature: {sig['id']}",
                detail=sig["pattern"],
                signature=sig,
            )
            self.kb.add_threat(ThreatRecord(
                id=f"sig-{sig['id']}",
                type="signature",
                name=sig["id"],
                description=sig["pattern"],
                severity="medium",
                source_module=self.MODULE_ID,
            ))

        result.metrics = {
            "techniques_analyzed": len(techniques),
            "tactics_observed": len(tactic_counts),
            "groups_attributed": len(attributed_groups),
            "signatures_generated": len(signatures),
            "kill_chain_stage": kill_chain_stage,
            "prediction_confidence": prediction.confidence if prediction else 0.0,
        }

        return self._finish_result(result, t0)

    def _estimate_kill_chain_stage(self, tactic_counts: Counter) -> int:
        """0-13 mapping to Lockheed Martin / MITRE kill chain stages."""
        stage_order = {t: i for i, t in enumerate(MITRE_TACTICS)}
        stages = [stage_order[t] for t in tactic_counts if t in stage_order]
        return max(stages) if stages else 0

    def _attribute_to_groups(self, techniques: list[str]) -> list[ThreatGroup]:
        tid_set = set(techniques)
        scored = []
        for group in KNOWN_GROUPS:
            overlap = len(tid_set & set(group.techniques))
            if overlap >= 1:
                scored.append((overlap / len(group.techniques), group))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in scored[:3]]

    def _predict_next(self, techniques: list[str]) -> TTPPrediction:
        """Predict next likely technique using LSTM or Markov fallback."""
        if _TORCH and self._predictor:
            return self._neural_predict(techniques)
        return self._markov_predict(techniques)

    def _neural_predict(self, techniques: list[str]) -> TTPPrediction:
        import torch
        ids = [self._technique_index.get(t, 0) for t in techniques[-8:]]
        x = torch.tensor([ids], dtype=torch.long).to(self._device)
        with torch.no_grad():
            logits = self._predictor(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        # Top-5 predictions
        all_techs = list(EMBEDDED_TECHNIQUES.keys())
        tech_probs = sorted(
            [(t, probs[self._technique_index.get(t, 0)]) for t in all_techs],
            key=lambda x: x[1], reverse=True
        )[:5]

        top_tech = tech_probs[0][0] if tech_probs else "T1059"
        tactic = EMBEDDED_TECHNIQUES.get(top_tech, {}).get("tactic", "unknown")

        stage_order = {t: i for i, t in enumerate(MITRE_TACTICS)}
        stage = stage_order.get(tactic, 0)

        return TTPPrediction(
            predicted_techniques=[t for t, _ in tech_probs],
            probabilities={t: round(p, 4) for t, p in tech_probs},
            next_likely_tactic=tactic,
            confidence=round(tech_probs[0][1] if tech_probs else 0.0, 4),
            kill_chain_position=stage,
        )

    def _markov_predict(self, techniques: list[str]) -> TTPPrediction:
        """Simple bigram Markov chain fallback."""
        # Compute transition from last observed technique
        last = techniques[-1] if techniques else "T1566"
        last_tactic = EMBEDDED_TECHNIQUES.get(last, {}).get("tactic", "initial-access")

        # Look at next tactic in kill chain
        stage_order = {t: i for i, t in enumerate(MITRE_TACTICS)}
        next_stage = min(stage_order.get(last_tactic, 0) + 1, len(MITRE_TACTICS) - 1)
        next_tactic = MITRE_TACTICS[next_stage]

        candidates = [t for t, info in EMBEDDED_TECHNIQUES.items()
                      if info["tactic"] == next_tactic]

        return TTPPrediction(
            predicted_techniques=candidates[:5],
            probabilities={t: 1.0 / max(len(candidates), 1) for t in candidates[:5]},
            next_likely_tactic=next_tactic,
            confidence=0.4,
            kill_chain_position=next_stage,
        )

    def _generate_signatures(self, techniques: list[str]) -> list[dict]:
        sigs = []
        for tid in techniques:
            tech = EMBEDDED_TECHNIQUES.get(tid)
            if not tech:
                continue
            # Generate Sigma-style detection rule skeleton
            sigs.append({
                "id": f"OLYMPUS-{tid}",
                "technique": tid,
                "technique_name": tech["name"],
                "tactic": tech["tactic"],
                "pattern": f"DETECT: {tech['name']} activity (MITRE {tid})",
                "format": "sigma",
                "logsource": "process_creation" if "execution" in tech["tactic"] else "network",
            })
        return sigs
